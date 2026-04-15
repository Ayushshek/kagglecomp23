I will add the solution to the competition when it ends on april 12th. I placed 253/1171 not bad for my first competition.


**Pipeline Overview**

This project is an end-to-end LightGBM pipeline for panel time-series regression (multiple entities over time), designed for competition-style training + submission generation.

It does four major things:

1. Loads train/test parquet datasets.
2. Builds leakage-safe lag/rolling features for all `feature_*` columns.
3. Tunes a targeted subset of LightGBM hyperparameters with expanding time-based CV.
4. Trains a final model, predicts test rows, writes a submission, and saves model artifacts.

Core behavior:

- Time structure is controlled by `ts_index`.
- Series grouping for temporal features is by `(code, sub_code)`.
- Recency weighting is applied by `sub_category` so newer rows get higher effective training weight.
- Categorical columns are train-fitted label encoded; unseen test categories become `-1`.
- Final feature set is restricted by a precomputed feature-importance whitelist (`feature_importance_gain_desc.csv`), top N rows.

---

**End-To-End Flow**

1. Read `train.parquet` and `test.parquet`.
2. Detect base features as all columns prefixed with `feature_`.
3. Downcast base numeric features to `float32` for memory efficiency.
4. Generate train temporal features:
   - Lags: `1, 3, 5, 14, 21, 28`
   - Rolling means: windows `3` and `7`
   - Rolling std: window `7`
   - Rolling stats are computed on `shift(1)` values to avoid target-time leakage.
5. Generate test temporal features using train history carry-over:
   - Keep last `MAX_HISTORY` rows per group from train.
   - Concatenate with test rows and compute same lag/rolling transforms.
   - Extract transformed values back to test rows only.
6. Encode categoricals: `code`, `sub_code`, `sub_category`, `horizon`.
7. Build model feature list:
   - Base features + generated temporal features + used categorical columns.
   - Remove banned columns (`id`, target, weight, `ts_index`).
   - Keep only features present in top-N whitelist order.
8. Hyperparameter step:
   - Either skip tuning and use defaults, or run Optuna stage-3 tuning.
   - Stage-3 tunes only regularization-like params:
     - `lambda_l1`, `lambda_l2`, `min_gain_to_split`
9. Train final LightGBM on all train rows.
10. Predict test rows.
11. Write `submission.csv` (`id,prediction`).
12. Save `model.txt` and `model.metadata.json`.

---

**Weighting Strategy**

Training uses combined weights:

- Base weight = dataset `weight` column if present, else `1`.
- Recency multiplier = `exp(-alpha * age)` where:
  - `age = max(ts_index within sub_category) - ts_index`
- Final training weight = `base_weight * recency_multiplier`

This biases fitting toward more recent observations within each `sub_category` while still respecting provided sample weights.

---

**Cross-Validation and Tuning Logic**

- CV is expanding-window over sorted unique timestamps.
- Each fold trains on earlier time and validates on later time.
- Minimum initial train span uses `MIN_TRAIN_RATIO` of unique timestamps.
- Tuning objective is weighted RMSE on validation folds.
- Early stopping and mean best iteration across folds determine final `num_boost_round`.
- Optuna storage can persist to SQLite for resumable tuning.

---

**Outputs**

- `submission.csv`: Kaggle-style predictions.
- `model.txt`: serialized LightGBM booster.
- `model.metadata.json`: params + features + categorical list + recency alpha + boosting rounds.
- Optional Optuna DB file for study persistence.

---

**File-By-File Explanation**

1. [main.py](/Users/ayushshekhar/Documents/k3/klate/s3/main.py)  
   - Orchestrates the full pipeline.
   - Parses CLI flags (paths, batch size, trials, splits, skip tuning, disable Optuna storage).
   - Loads feature whitelist from CSV and validates it.
   - Calls feature engineering for train/test.
   - Encodes categoricals.
   - Selects final model feature columns in whitelist order.
   - Runs tuning or fallback defaults.
   - Trains final model, predicts, writes submission, saves artifacts.

2. [config.py](/Users/ayushshekhar/Documents/k3/klate/s3/config.py)  
   - Centralized constants and paths.
   - Defines dataset columns, categorical columns, objective/metric, learning rates, CV settings, and random seed.
   - Defines temporal feature recipe (lags/rolling windows), dtype/downcast settings, and history size.
   - Defines Optuna search space and fixed LightGBM parameters.
   - Defines output paths (`submission.csv`, `model.txt`, `optuna db`, feature-importance file).

3. [feature_engineering.py](/Users/ayushshekhar/Documents/k3/klate/s3/feature_engineering.py)  
   - Memory-aware temporal feature generation module.
   - Finds base feature columns and enumerates generated feature names.
   - Downcasts base features to reduce memory.
   - Computes lag/rolling blocks in batches (`FEATURE_BATCH_SIZE`) to avoid large peak RAM.
   - `add_time_series_features_train`:
     - Sorts by group + time.
     - Creates lag and rolling features grouped by `(code, sub_code)`.
   - `add_time_series_features_test`:
     - Uses tail train history + test rows to compute valid temporal features for early test rows.
   - Includes `build_row_key` utility for constructing a composite unique row identifier string.

4. [training.py](/Users/ayushshekhar/Documents/k3/klate/s3/training.py)  
   - Final model training and artifact persistence.
   - `SafeLabelEncoder`:
     - Fits mappings on train categories only.
     - Maps unknown values to `-1` on transform.
   - `encode_categoricals` applies consistent encoding to train/test.
   - `apply_recency_training_weights` combines dataset weights with recency weights.
   - `get_default_lgb_params` provides non-tuned fallback params.
   - `train_final_model` builds LightGBM dataset and trains final booster.
   - `save_model_and_metadata` writes model and JSON metadata.

5. [optuna_tuning.py](/Users/ayushshekhar/Documents/k3/klate/s3/optuna_tuning.py)  
   - Stage-3-only hyperparameter tuning logic.
   - Implements:
     - Weighted RMSE objective.
     - Recency weight computation.
     - Expanding-window fold construction for time-series CV.
   - Builds Optuna study with optional SQLite storage and resumable trial counting.
   - Tunes only selected regularization/split-gain parameters while keeping core tree params fixed.
   - Captures fold-level early-stopped best iterations and returns averaged value for final training rounds.
   - Returns `TuningResult` with best params, recency alpha, boosting rounds, and summary data.

6. [inference.py](/Users/ayushshekhar/Documents/k3/klate/s3/inference.py)  
   - Lightweight inference/output utilities.
   - `predict_test` runs booster prediction on selected feature columns.
   - `write_submission` validates presence of `id`, writes `id,prediction` CSV, and ensures output directory exists.

7. [feature_importance_gain_desc.csv](/Users/ayushshekhar/Documents/k3/klate/s3/feature_importance_gain_desc.csv)  
   - Ordered feature-importance list used as a whitelist.
   - Pipeline reads the top `FEATURE_IMPORTANCE_TOP_N` rows and uses the `feature` column to constrain model features.
   - This stabilizes feature selection and enforces a fixed feature order across runs.

---

If you want, I can turn this into a polished `README.md` draft with sections like Installation, Data Schema, CLI Usage, and Reproducibility Notes next.
