"""Inference and submission helpers."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from config import ID_COL, PREDICTION_COL, SUBMISSION_PATH


def predict_test(model: lgb.Booster, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Run model inference on test features."""
    preds = model.predict(test_df[feature_cols])
    return pd.Series(preds, index=test_df.index, name=PREDICTION_COL)


def write_submission(
    test_df: pd.DataFrame,
    predictions: pd.Series,
    output_path: Path = SUBMISSION_PATH,
) -> Path:
    """Write Kaggle submission file in id,prediction format."""
    if ID_COL not in test_df.columns:
        raise ValueError(f"Missing id column '{ID_COL}' in test data.")

    submission = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL].values,
            PREDICTION_COL: predictions.values,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path
