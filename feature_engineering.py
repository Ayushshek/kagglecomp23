"""Memory-efficient time-series feature generation."""

from __future__ import annotations

import gc
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from config import (
    FEATURE_BATCH_SIZE,
    FEATURE_DTYPE,
    FEATURE_PREFIX,
    GROUP_COLS,
    LAG_STEPS,
    MAX_HISTORY,
    ROLLING_WINDOWS,
    TS_INDEX_COL,
)


def get_base_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return base feature columns (feature_* only), sorted for reproducibility."""
    return sorted([col for col in df.columns if col.startswith(FEATURE_PREFIX)])


def generated_feature_names(base_feature_cols: Sequence[str]) -> list[str]:
    """Return all lag/rolling feature names generated from base features."""
    names: list[str] = []
    for col in base_feature_cols:
        for lag in LAG_STEPS:
            names.append(f"{col}_lag_{lag}")
        for roll_name in ROLLING_WINDOWS:
            names.append(f"{col}_{roll_name}")
    return names


def downcast_base_features(df: pd.DataFrame, base_feature_cols: Sequence[str]) -> None:
    """Downcast base numeric features in-place to reduce memory footprint."""
    for col in base_feature_cols:
        if col in df.columns and df[col].dtype != FEATURE_DTYPE:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(FEATURE_DTYPE)


def _iter_batches(values: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    if batch_size <= 0:
        yield list(values)
        return
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def _compute_feature_block(
    sorted_batch: pd.DataFrame,
    group_keys: list[pd.Series],
) -> pd.DataFrame:
    """Compute lag and rolling features for one feature batch in sorted order."""
    grouped = sorted_batch.groupby(group_keys, sort=False, observed=True)
    out_blocks: list[pd.DataFrame] = []

    for lag in LAG_STEPS:
        lagged = grouped.shift(lag)
        lagged.columns = [f"{c}_lag_{lag}" for c in sorted_batch.columns]
        out_blocks.append(lagged)

    shifted = grouped.shift(1)

    for roll_name, window in ROLLING_WINDOWS.items():
        rolling_group = shifted.groupby(group_keys, sort=False, observed=True)
        if roll_name == "rolling_std_7":
            rolled = (
                rolling_group.rolling(window, min_periods=window)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )
        else:
            rolled = (
                rolling_group.rolling(window, min_periods=window)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
        rolled.columns = [f"{c}_{roll_name}" for c in sorted_batch.columns]
        out_blocks.append(rolled)

    block = pd.concat(out_blocks, axis=1)
    return block.astype(FEATURE_DTYPE)


def _validate_inputs(df: pd.DataFrame, base_feature_cols: Sequence[str]) -> None:
    required = set(GROUP_COLS + [TS_INDEX_COL])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    missing_base = [col for col in base_feature_cols if col not in df.columns]
    if missing_base:
        raise ValueError(f"Missing base features: {missing_base[:5]}")


def add_time_series_features_train(
    train_df: pd.DataFrame,
    base_feature_cols: Sequence[str],
    batch_size: int = FEATURE_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Add lag/rolling features to training data in memory-conscious batches.

    Grouping is strictly by (code, sub_code), sorted by ts_index.
    """
    _validate_inputs(train_df, base_feature_cols)

    sort_cols = GROUP_COLS + [TS_INDEX_COL]
    sorted_index = train_df.sort_values(sort_cols, kind="mergesort").index
    group_keys = [train_df.loc[sorted_index, c] for c in GROUP_COLS]

    for batch_cols in _iter_batches(base_feature_cols, batch_size):
        sorted_batch = train_df.loc[sorted_index, batch_cols]
        block_sorted = _compute_feature_block(sorted_batch, group_keys)
        block_aligned = block_sorted.reindex(train_df.index)
        train_df = pd.concat([train_df, block_aligned], axis=1, copy=False)

        del sorted_batch, block_sorted, block_aligned
        gc.collect()

    return train_df


def add_time_series_features_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_feature_cols: Sequence[str],
    batch_size: int = FEATURE_BATCH_SIZE,
    history_len: int = MAX_HISTORY,
) -> pd.DataFrame:
    """
    Add lag/rolling features to test using train-history carry-over.

    Early test rows in each (code, sub_code) group are prefixed with the last
    available train history so lag/rolling features can be computed.
    """
    _validate_inputs(train_df, base_feature_cols)
    _validate_inputs(test_df, base_feature_cols)

    cols_needed = list(dict.fromkeys(GROUP_COLS + [TS_INDEX_COL] + list(base_feature_cols)))

    train_tail = (
        train_df.loc[:, cols_needed]
        .sort_values(GROUP_COLS + [TS_INDEX_COL], kind="mergesort")
        .groupby(GROUP_COLS, sort=False, observed=True)
        .tail(history_len)
        .copy()
    )
    train_tail["__is_test"] = 0
    train_tail["__test_row_id"] = -1

    test_work = test_df.loc[:, cols_needed].copy()
    test_work["__is_test"] = 1
    test_work["__test_row_id"] = np.arange(len(test_work), dtype=np.int64)

    combined = pd.concat([train_tail, test_work], axis=0, ignore_index=True, copy=False)

    sort_cols = GROUP_COLS + ["__is_test", TS_INDEX_COL, "__test_row_id"]
    sorted_index = combined.sort_values(sort_cols, kind="mergesort").index
    group_keys = [combined.loc[sorted_index, c] for c in GROUP_COLS]
    test_mask = combined["__is_test"].eq(1)
    test_row_ids = combined.loc[test_mask, "__test_row_id"].to_numpy()

    for batch_cols in _iter_batches(base_feature_cols, batch_size):
        sorted_batch = combined.loc[sorted_index, batch_cols]
        block_sorted = _compute_feature_block(sorted_batch, group_keys)
        block_aligned = block_sorted.reindex(combined.index)

        test_block = block_aligned.loc[test_mask].copy()
        test_block.index = test_row_ids
        test_block = test_block.sort_index(kind="mergesort")
        test_block.index = test_df.index
        test_df = pd.concat([test_df, test_block], axis=1, copy=False)

        del sorted_batch, block_sorted, block_aligned, test_block
        gc.collect()

    del train_tail, test_work, combined
    gc.collect()

    return test_df


def build_row_key(df: pd.DataFrame) -> pd.Series:
    """Create the unique row identifier string from competition key columns."""
    required = ["code", "sub_code", "sub_category", "horizon", "ts_index"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for row-key construction: {missing}")

    return (
        df["code"].astype(str)
        + "__"
        + df["sub_code"].astype(str)
        + "__"
        + df["sub_category"].astype(str)
        + "__"
        + df["horizon"].astype(str)
        + "__"
        + df["ts_index"].astype(str)
    )
