"""End-to-end LightGBM time-series training + submission pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_COLS,
    FEATURE_BATCH_SIZE,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_TOP_N,
    N_CV_SPLITS,
    N_OPTUNA_TRIALS,
    OPTUNA_DB_PATH,
    RECENCY_ALPHA,
    SUBMISSION_PATH,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    TS_INDEX_COL,
    WEIGHT_COL,
)
from feature_engineering import (
    add_time_series_features_test,
    add_time_series_features_train,
    downcast_base_features,
    generated_feature_names,
    get_base_feature_columns,
)
from inference import predict_test, write_submission
from optuna_tuning import TuningResult, run_staged_optuna
from training import (
    ModelArtifacts,
    encode_categoricals,
    get_default_lgb_params,
    save_model_and_metadata,
    train_final_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged LightGBM time-series pipeline")
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--submission-path", type=Path, default=SUBMISSION_PATH)
    parser.add_argument("--batch-size", type=int, default=FEATURE_BATCH_SIZE)
    parser.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument("--n-splits", type=int, default=N_CV_SPLITS)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--no-optuna-storage", action="store_true")
    return parser.parse_args()


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def select_feature_columns(
    train_df: pd.DataFrame,
    base_feature_cols: list[str],
    categorical_cols: list[str],
    ordered_feature_whitelist: list[str],
) -> list[str]:
    generated_cols = generated_feature_names(base_feature_cols)

    feature_cols: list[str] = []
    feature_cols.extend(base_feature_cols)
    feature_cols.extend([c for c in generated_cols if c in train_df.columns])
    feature_cols.extend([c for c in categorical_cols if c in train_df.columns])

    deduped = list(dict.fromkeys(feature_cols))
    banned = {TARGET_COL, WEIGHT_COL, TS_INDEX_COL, "id"}
    return [c for c in ordered_feature_whitelist if c in deduped and c not in banned]


def load_feature_whitelist(path: Path, top_n: int) -> list[str]:
    if top_n <= 0:
        raise ValueError(f"feature whitelist top_n must be positive; got {top_n}")

    try:
        importance_df = pd.read_csv(path, usecols=["feature"])
    except ValueError as exc:
        raise ValueError(f"Feature-importance file must include a 'feature' column: {path}") from exc

    top_rows = importance_df.head(top_n)
    ordered = top_rows["feature"].dropna().astype(str).tolist()

    deduped_ordered = list(dict.fromkeys(ordered))
    if not deduped_ordered:
        raise ValueError(f"No feature names found in top {top_n} rows of: {path}")
    return deduped_ordered


def get_tuning_result(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    args: argparse.Namespace,
) -> TuningResult:
    if args.skip_tuning:
        return TuningResult(
            best_params=get_default_lgb_params(),
            best_recency_alpha=RECENCY_ALPHA,
            best_num_boost_round=1500,
            stage_summaries={"status": {"message": "Optuna tuning skipped"}},
        )

    storage_path = None if args.no_optuna_storage else OPTUNA_DB_PATH
    return run_staged_optuna(
        train_df=train_df,
        feature_cols=feature_cols,
        categorical_cols=[c for c in categorical_cols if c in feature_cols],
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        storage_path=storage_path,
    )


def main() -> None:
    np.random.seed(42)
    args = parse_args()
    ordered_feature_whitelist = load_feature_whitelist(FEATURE_IMPORTANCE_PATH, FEATURE_IMPORTANCE_TOP_N)

    train_df, test_df = load_data(args.train_path, args.test_path)

    base_feature_cols = get_base_feature_columns(train_df)
    if not base_feature_cols:
        raise ValueError("No base feature columns found (expected columns starting with 'feature_').")

    downcast_base_features(train_df, base_feature_cols)
    downcast_base_features(test_df, base_feature_cols)

    train_df = add_time_series_features_train(
        train_df=train_df,
        base_feature_cols=base_feature_cols,
        batch_size=args.batch_size,
    )
    test_df = add_time_series_features_test(
        train_df=train_df,
        test_df=test_df,
        base_feature_cols=base_feature_cols,
        batch_size=args.batch_size,
    )

    train_df, test_df, _, used_categorical_cols = encode_categoricals(
        train_df=train_df,
        test_df=test_df,
        categorical_cols=CATEGORICAL_COLS,
    )

    feature_cols = select_feature_columns(
        train_df,
        base_feature_cols,
        used_categorical_cols,
        ordered_feature_whitelist=ordered_feature_whitelist,
    )
    if not feature_cols:
        raise ValueError(
            "No usable model features remained after applying top feature whitelist. "
            "Check feature_importance_gain_desc.csv and engineered feature creation."
        )

    tuning_result = get_tuning_result(train_df, feature_cols, used_categorical_cols, args)

    model = train_final_model(
        train_df=train_df,
        feature_cols=feature_cols,
        categorical_cols=used_categorical_cols,
        params=tuning_result.best_params,
        recency_alpha=tuning_result.best_recency_alpha,
        num_boost_round=tuning_result.best_num_boost_round,
    )

    preds = predict_test(model=model, test_df=test_df, feature_cols=feature_cols)
    submission_path = write_submission(
        test_df=test_df,
        predictions=preds,
        output_path=args.submission_path,
    )

    artifacts = ModelArtifacts(
        params=tuning_result.best_params,
        feature_cols=feature_cols,
        categorical_cols=used_categorical_cols,
        recency_alpha=tuning_result.best_recency_alpha,
        num_boost_round=tuning_result.best_num_boost_round,
    )
    save_model_and_metadata(model=model, artifacts=artifacts)

    # Keep outputs accessible for downstream scripting.
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
