"""Stage-3-only Optuna tuning for LightGBM time-series regression."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from config import (
    EARLY_STOPPING_ROUNDS,
    FIXED_LGB_PARAMS,
    HYPERPARAM_SPACE_STAGE3,
    LEARNING_RATE,
    METRIC,
    MIN_TRAIN_RATIO,
    N_CV_SPLITS,
    N_JOBS,
    N_OPTUNA_TRIALS,
    OBJECTIVE,
    OPTUNA_DB_PATH,
    RECENCY_ALPHA,
    RECENCY_GROUP_COL,
    SEED,
    TARGET_COL,
    TS_INDEX_COL,
    TUNING_LEARNING_RATE,
    TUNING_MAX_BOOST_ROUNDS,
    VERBOSITY,
    WEIGHT_COL,
)


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_recency_alpha: float
    best_num_boost_round: int
    stage_summaries: dict[str, dict[str, Any]]


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Weighted RMSE used as Optuna objective."""
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    if float(w.sum()) == 0.0:
        w = np.ones_like(w, dtype=np.float64)
    mse = np.average((y_true - y_pred) ** 2, weights=w)
    return float(np.sqrt(mse))


def compute_recency_weights(
    ts_index: pd.Series,
    recency_group: pd.Series,
    alpha: float,
) -> np.ndarray:
    """Compute exp(-alpha * age), where age is group-local latest_ts - ts."""
    ts = ts_index.reset_index(drop=True)
    grp = recency_group.reset_index(drop=True)
    latest_per_row = ts.groupby(grp, observed=True).transform("max")
    age = latest_per_row - ts
    return np.exp(-alpha * age.to_numpy(dtype=np.float64))


def build_time_series_folds(
    ts_index: pd.Series,
    n_splits: int = N_CV_SPLITS,
    min_train_ratio: float = MIN_TRAIN_RATIO,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window folds using earlier timestamps for train, later for valid."""
    ts = ts_index.reset_index(drop=True)
    unique_ts = np.sort(ts.unique())
    n_unique = len(unique_ts)

    min_train_steps = max(1, int(n_unique * min_train_ratio))
    remaining = n_unique - min_train_steps
    step = remaining // (n_splits + 1)

    if step < 1:
        raise ValueError(
            f"Not enough unique timestamps ({n_unique}) for {n_splits} splits."
        )

    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold in range(n_splits):
        train_end_idx = min_train_steps + (fold + 1) * step
        train_end_ts = unique_ts[train_end_idx - 1]

        if fold < n_splits - 1:
            valid_end_idx = min_train_steps + (fold + 2) * step
            valid_end_ts = unique_ts[valid_end_idx - 1]
        else:
            valid_end_ts = unique_ts[-1]

        train_mask = ts <= train_end_ts
        valid_mask = (ts > train_end_ts) & (ts <= valid_end_ts)

        train_idx = np.where(train_mask)[0]
        valid_idx = np.where(valid_mask)[0]

        if len(train_idx) > 0 and len(valid_idx) > 0:
            folds.append((train_idx, valid_idx))

    if not folds:
        raise ValueError("No valid time-series folds were generated.")

    return folds


def _base_lgb_params(learning_rate: float) -> dict[str, Any]:
    return {
        "objective": OBJECTIVE,
        "metric": METRIC,
        "learning_rate": learning_rate,
        "verbosity": VERBOSITY,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "n_jobs": N_JOBS,
    }


def _suggest_value(trial: optuna.Trial, name: str, bounds: tuple[float, float]) -> Any:
    low, high = bounds
    int_params = {"num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq"}
    if name in int_params:
        return trial.suggest_int(name, int(low), int(high))
    return trial.suggest_float(name, float(low), float(high))


def _make_stage3_objective(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    fixed_params: dict[str, Any],
    fixed_recency_alpha: float,
) -> Callable[[optuna.Trial], float]:
    X = df[feature_cols]
    y = df[TARGET_COL]
    ts = df[TS_INDEX_COL]

    if RECENCY_GROUP_COL not in df.columns:
        raise ValueError(f"Missing recency group column: {RECENCY_GROUP_COL}")

    recency_group = df[RECENCY_GROUP_COL]
    dataset_weights = (
        df[WEIGHT_COL].astype("float64")
        if WEIGHT_COL in df.columns
        else pd.Series(np.ones(len(df), dtype=np.float64), index=df.index)
    )

    def objective(trial: optuna.Trial) -> float:
        params = dict(fixed_params)
        for name, bounds in HYPERPARAM_SPACE_STAGE3.items():
            params[name] = _suggest_value(trial, name, bounds)

        fold_scores: list[float] = []
        fold_iters: list[int] = []

        for train_idx, valid_idx in folds:
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            w_train = dataset_weights.iloc[train_idx].to_numpy(copy=True)
            w_valid = dataset_weights.iloc[valid_idx].to_numpy(copy=False)

            recency_train = compute_recency_weights(
                ts.iloc[train_idx],
                recency_group.iloc[train_idx],
                fixed_recency_alpha,
            )
            w_train *= recency_train

            train_set = lgb.Dataset(
                X_train,
                label=y_train,
                weight=w_train,
                categorical_feature=categorical_cols,
                free_raw_data=False,
            )
            valid_set = lgb.Dataset(
                X_valid,
                label=y_valid,
                weight=w_valid,
                reference=train_set,
                free_raw_data=False,
            )

            try:
                model = lgb.train(
                    params,
                    train_set,
                    num_boost_round=TUNING_MAX_BOOST_ROUNDS,
                    valid_sets=[valid_set],
                    callbacks=[
                        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
            except lgb.basic.LightGBMError:
                return float("inf")

            pred = model.predict(X_valid, num_iteration=model.best_iteration)
            fold_scores.append(weighted_rmse(y_valid.to_numpy(), pred, w_valid))
            fold_iters.append(int(model.best_iteration))

        trial.set_user_attr("mean_best_iteration", int(np.mean(fold_iters)))
        trial.set_user_attr("recency_alpha", fixed_recency_alpha)

        return float(np.mean(fold_scores))

    return objective


def _make_study(name: str, sampler: TPESampler, storage_path: Path | None) -> optuna.Study:
    if storage_path is None:
        return optuna.create_study(direction="minimize", study_name=name, sampler=sampler)

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}"
    return optuna.create_study(
        direction="minimize",
        study_name=name,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )


def _signature(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _remaining_trials(study: optuna.Study, target_trials: int) -> int:
    completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    return max(0, target_trials - completed)


def run_staged_optuna(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    n_trials: int = N_OPTUNA_TRIALS,
    n_splits: int = N_CV_SPLITS,
    tuning_learning_rate: float = TUNING_LEARNING_RATE,
    storage_path: Path | None = OPTUNA_DB_PATH,
) -> TuningResult:
    """Run stage-3-only Optuna and return final params + fixed recency alpha + boosting rounds."""
    folds = build_time_series_folds(train_df[TS_INDEX_COL], n_splits=n_splits)
    sampler = TPESampler(seed=SEED)

    fixed_params = _base_lgb_params(tuning_learning_rate)
    fixed_params.update(FIXED_LGB_PARAMS)

    stage3_sig = _signature(
        {
            "fixed_params": fixed_params,
            "recency_alpha": RECENCY_ALPHA,
            "search_space": HYPERPARAM_SPACE_STAGE3,
        }
    )
    stage3_study = _make_study(f"stage3_regularization_only_{stage3_sig}", sampler, storage_path)
    stage3_obj = _make_stage3_objective(
        df=train_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        folds=folds,
        fixed_params=fixed_params,
        fixed_recency_alpha=RECENCY_ALPHA,
    )
    rem = _remaining_trials(stage3_study, n_trials)
    if rem > 0:
        stage3_study.optimize(stage3_obj, n_trials=rem)

    final_tuned_params = dict(fixed_params)
    final_tuned_params.update(stage3_study.best_params)
    final_tuned_params["learning_rate"] = LEARNING_RATE

    best_num_boost_round = int(
        stage3_study.best_trial.user_attrs.get("mean_best_iteration", 1000)
    )

    summary = {
        "stage3": {
            "trials": n_trials,
            "best_value": stage3_study.best_value,
            "best_params": stage3_study.best_params,
            "fixed_params": FIXED_LGB_PARAMS,
            "fixed_recency_alpha": RECENCY_ALPHA,
        }
    }

    return TuningResult(
        best_params=final_tuned_params,
        best_recency_alpha=RECENCY_ALPHA,
        best_num_boost_round=best_num_boost_round,
        stage_summaries=summary,
    )
