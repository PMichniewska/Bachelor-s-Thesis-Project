"""
XGBoost regression training script.
Trains on train_dataset.parquet, evaluates on test_dataset.parquet.
Switch DATA_MODE to "sample" to run on dataset_sample.parquet for quick testing.

GPU usage: set DEVICE = "cuda" (requires XGBoost built with CUDA support).
CPU usage: set DEVICE = "cpu".
"""

import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- Config ----------
TARGET     = "bpm"
DROP_COLS  = ["file_name", "frame_idx"]   # non-feature columns
DEVICE     = "cuda"                        # "cuda" for GPU, "cpu" for CPU
RANDOM_SEED = 42

PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "device":           DEVICE,
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state":     RANDOM_SEED,
}

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",  default=".",         help="Directory containing parquet files")
parser.add_argument("--out_dir",   default="outputs",   help="Directory for results and model")
parser.add_argument("--mode",      default="full",      choices=["full", "sample"],
                    help="'full' uses train/test parquets; 'sample' uses dataset_sample.parquet")
parser.add_argument("--device",    default=DEVICE,      choices=["cuda", "cpu"])
args = parser.parse_args()

PARAMS["device"] = args.device
os.makedirs(args.out_dir, exist_ok=True)

# ---------- Load ----------
if args.mode == "sample":
    print("Mode: sample  (dataset_sample.parquet, 80/20 row split)")
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train_dataset_sample.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.data_dir, "test_dataset_sample.parquet"))
    train_df = train_df.drop(columns=DROP_COLS, errors="ignore")
    test_df  = test_df.drop(columns=DROP_COLS, errors="ignore")
else:
    print("Mode: full  (train_dataset.parquet / test_dataset.parquet)")
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train_dataset.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.data_dir, "test_dataset.parquet"))
    train_df = train_df.drop(columns=DROP_COLS, errors="ignore")
    test_df  = test_df.drop(columns=DROP_COLS, errors="ignore")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"Device: {PARAMS['device']}")

# ---------- Train ----------
model = xgb.XGBRegressor(**PARAMS, early_stopping_rounds=20, verbosity=1)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ---------- Evaluate ----------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\n--- Test set results ---")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.6f}")

# ---------- Save ----------
model_path = os.path.join(args.out_dir, "xgboost_model.ubj")
model.save_model(model_path)
print(f"\nModel saved : {model_path}")

results = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
results_path = os.path.join(args.out_dir, "xgboost_predictions.csv")
results.to_csv(results_path, index=False)
print(f"Predictions : {results_path}")
