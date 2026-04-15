"""Model training utilities for the LightGBM pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_COLS,
    FIXED_LGB_PARAMS,
    LEARNING_RATE,
    METRIC,
    MODEL_OUTPUT_PATH,
    N_JOBS,
    OBJECTIVE,
    RECENCY_ALPHA,
    RECENCY_GROUP_COL,
    SEED,
    TARGET_COL,
    TS_INDEX_COL,
    VERBOSITY,
    WEIGHT_COL,
)
from optuna_tuning import compute_recency_weights


class SafeLabelEncoder:
    """Train-only label encoder; unseen categories map to -1."""

    def __init__(self) -> None:
        self.mappings: dict[str, dict[Any, int]] = {}

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "SafeLabelEncoder":
        self.mappings = {}
        for col in cols:
            if col not in df.columns:
                continue
            values = pd.Series(df[col].dropna().unique())
            try:
                values = values.sort_values(ignore_index=True)
            except TypeError:
                values = values.reset_index(drop=True)
            self.mappings[col] = {val: int(i) for i, val in enumerate(values)}
        return self

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            if col not in df.columns or col not in self.mappings:
                continue
            mapping = self.mappings[col]
            df[col] = df[col].map(lambda x: mapping.get(x, -1)).astype("int32")
        return df

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return self.fit(df, cols).transform(df, cols)


@dataclass
class ModelArtifacts:
    params: dict[str, Any]
    feature_cols: list[str]
    categorical_cols: list[str]
    recency_alpha: float
    num_boost_round: int


def get_default_lgb_params() -> dict[str, Any]:
    """Fallback params used if tuning is skipped."""
    params: dict[str, Any] = {
        "objective": OBJECTIVE,
        "metric": METRIC,
        "learning_rate": LEARNING_RATE,
        "verbosity": VERBOSITY,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "n_jobs": N_JOBS,
    }
    params.update(FIXED_LGB_PARAMS)
    params.update(
        {
            "lambda_l1": 0.4882285122821464,
            "lambda_l2": 1.4607232426760908,
            "min_gain_to_split": 0.18318092164684585,
        }
    )
    return params


def apply_recency_training_weights(
    df: pd.DataFrame,
    alpha: float = RECENCY_ALPHA,
) -> np.ndarray:
    """Compute final train weights = dataset_weight * recency_weight."""
    if RECENCY_GROUP_COL not in df.columns:
        raise ValueError(f"Missing recency group column: {RECENCY_GROUP_COL}")

    base_weights = (
        df[WEIGHT_COL].to_numpy(dtype=np.float64, copy=False)
        if WEIGHT_COL in df.columns
        else np.ones(len(df), dtype=np.float64)
    )

    recency = compute_recency_weights(
        df[TS_INDEX_COL],
        df[RECENCY_GROUP_COL],
        alpha,
    )
    return base_weights * recency


def train_final_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict[str, Any],
    recency_alpha: float,
    num_boost_round: int,
) -> lgb.Booster:
    """Train final LightGBM model on all training data using tuned boosting rounds."""
    X = train_df[feature_cols]
    y = train_df[TARGET_COL]
    w = apply_recency_training_weights(train_df, alpha=recency_alpha)

    train_set = lgb.Dataset(
        X,
        label=y,
        weight=w,
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=int(num_boost_round),
    )
    return model


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SafeLabelEncoder, list[str]]:
    """Fit encoder on train only and transform both train/test."""
    cols = categorical_cols or CATEGORICAL_COLS
    encoder = SafeLabelEncoder().fit(train_df, cols)
    train_df = encoder.transform(train_df, cols)
    test_df = encoder.transform(test_df, cols)
    used_cols = [c for c in cols if c in train_df.columns and c in test_df.columns]
    return train_df, test_df, encoder, used_cols


def save_model_and_metadata(
    model: lgb.Booster,
    artifacts: ModelArtifacts,
    output_model_path: Path = MODEL_OUTPUT_PATH,
) -> None:
    """Persist model file and metadata JSON."""
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_model_path))

    metadata_path = output_model_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(asdict(artifacts), fp, indent=2)
