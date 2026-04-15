"""Configuration for the LightGBM time-series pipeline."""

from __future__ import annotations

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.parquet"
TEST_PATH = BASE_DIR / "test.parquet"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "model.txt"
OPTUNA_DB_PATH = BASE_DIR / "optuna_stage3_only.db"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "feature_importance_gain_desc.csv"

# Core columns
ID_COL = "id"
TARGET_COL = "y_target"
WEIGHT_COL = "weight"
TS_INDEX_COL = "ts_index"
GROUP_COLS = ["code", "sub_code"]
RECENCY_GROUP_COL = "sub_category"
CATEGORICAL_COLS = ["code", "sub_code", "sub_category", "horizon"]

# Model constants
LEARNING_RATE = 0.01
TUNING_LEARNING_RATE = 0.01
OBJECTIVE = "regression"
METRIC = "rmse"
EARLY_STOPPING_ROUNDS = 50
VERBOSITY = -1
TUNING_MAX_BOOST_ROUNDS = 5000

# Recency weighting
RECENCY_ALPHA = 0.00012481516711396667

# Optuna setup
N_OPTUNA_TRIALS = 60

HYPERPARAM_SPACE_STAGE3 = {
    "lambda_l1": (0.1, 0.8),
    "lambda_l2": (0.8, 1.8),
    "min_gain_to_split": (0.02, 0.08),
}

# Fixed non-stage3 parameters for stage3-only tuning.
FIXED_LGB_PARAMS = {
    "num_leaves": 401,
    "max_depth": 10,
    "min_data_in_leaf": 1059,
    "feature_fraction": 0.877728168328829,
    "bagging_fraction": 0.8954376941646838,
    "bagging_freq": 4,
}

FEATURE_IMPORTANCE_TOP_N = 701

# Feature engineering
LAG_STEPS = [1, 3, 5, 14, 21, 28]
ROLLING_WINDOWS = {
    "rolling_mean_3": 3,
    "rolling_mean_7": 7,
    "rolling_std_7": 7,
}
FEATURE_PREFIX = "feature_"
FEATURE_BATCH_SIZE = 8
FEATURE_DTYPE = "float32"
MAX_HISTORY = max(max(LAG_STEPS), max(ROLLING_WINDOWS.values()))

# Validation and reproducibility
N_CV_SPLITS = 5
MIN_TRAIN_RATIO = 0.5
SEED = 42
N_JOBS = -1

# Submission
PREDICTION_COL = "prediction"
