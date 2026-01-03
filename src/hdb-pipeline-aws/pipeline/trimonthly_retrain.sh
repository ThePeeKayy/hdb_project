#!/bin/bash
set -euo pipefail

PIPELINE_DIR="/home/ubuntu/pipeline"
LOG_DIR="/home/ubuntu/logs"
LOG_FILE="$LOG_DIR/trimonthly_retrain_$(date +%Y%m%d_%H%M%S).log"

S3_BUCKET="hdb-prediction-pipeline"
SILVER_KEY="silver/features.parquet"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "TRIMONTHLY RETRAIN (FROM SILVER features.parquet)"
log "=========================================="

cd "$PIPELINE_DIR"

# cron-safe conda activation
source /home/ubuntu/miniconda3/bin/activate hdb-env

log "→ Running monthly pipeline first..."
bash "$PIPELINE_DIR/monthly.sh" >> "$LOG_FILE" 2>&1
log "✓ Monthly pipeline completed"

log "→ Downloading silver/features.parquet from S3..."
aws s3 cp "s3://$S3_BUCKET/$SILVER_KEY" "$PIPELINE_DIR/features.parquet" >> "$LOG_FILE" 2>&1
log "✓ features.parquet downloaded"

log "→ Training SARIMAX models from features.parquet (inline python)..."
python3 - <<'PY' >> "$LOG_FILE" 2>&1
import json, gzip, pickle, logging
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trimonthly_retrain")

FEATURES_PATH = "features.parquet"
PKL_PATH = "sarimax-models-dict.pkl"
PKL_GZ_PATH = "sarimax-models-dict.pkl.gz"
METRICS_PATH = "model-metrics.json"

# --- Load data (must match gold_predictions expectations: month, group_key, avg_price) ---
df = pd.read_parquet(FEATURES_PATH)

required = {"group_key", "month", "avg_price"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"features.parquet missing required columns: {sorted(missing)}")

df = df.copy()
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.dropna(subset=["month", "group_key", "avg_price"])
df = df.sort_values(["group_key", "month"])

logger.info(f"Loaded {len(df):,} rows | groups={df['group_key'].nunique():,} | "
            f"range={df['month'].min()}..{df['month'].max()}")

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

models_dict = {}
individual_metrics = []

# Training configuration
MIN_MONTHS = 50
VALIDATION_MONTHS = 5

# Get feature columns (exclude core columns)
feature_cols = [col for col in df.columns if col not in ["month", "region", "flat_type", "avg_price", "group_key"]]

for gk, gdf in df.groupby("group_key", sort=True):
    try:
        ts = gdf.set_index("month")["avg_price"].astype(float).sort_index()
        ts = ts.asfreq("MS")
        ts = ts.interpolate(method="time")

        if ts.shape[0] < MIN_MONTHS + VALIDATION_MONTHS:
            logger.info(f"Skipping {gk}: insufficient history ({ts.shape[0]} < {MIN_MONTHS + VALIDATION_MONTHS})")
            continue

        # simple train/val split: last 2 months for validation
        train = ts.iloc[:-VALIDATION_MONTHS]
        val = ts.iloc[-VALIDATION_MONTHS:]

        # SARIMAX config
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )

        fitted = model.fit(disp=False, maxiter=200)

        # validate (2-step forecast on held-out)
        fc = fitted.get_forecast(steps=VALIDATION_MONTHS).predicted_mean
        
        val_rmse = rmse(val.values, fc.values)
        val_mae = mae(val.values, fc.values)
        val_mape = mape(val.values, fc.values)

        # refit on full series (so saved model uses all available data)
        model_full = SARIMAX(
            ts,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fitted_full = model_full.fit(disp=False, maxiter=200)

        models_dict[gk] = {
            "model": fitted_full,
            "last_date": ts.index.max(),  # gold_predictions expects this
        }

        individual_metrics.append({
            "group": gk,
            "rmse": val_rmse,
            "mae": val_mae,
            "mape": val_mape,
            "training_months": int(len(train))
        })

        logger.info(f"Trained {gk}: RMSE={val_rmse:.2f}, MAE={val_mae:.2f}, MAPE={val_mape:.2f}%")

    except Exception as e:
        logger.warning(f"Failed to train {gk}: {str(e)}")

if not models_dict:
    raise RuntimeError("No models trained (all groups skipped/failed).")

# Calculate average metrics
avg_rmse = np.mean([m["rmse"] for m in individual_metrics])
avg_mae = np.mean([m["mae"] for m in individual_metrics])
avg_mape = np.mean([m["mape"] for m in individual_metrics])

# Build comprehensive metrics JSON (matching testmodel.py format)
metrics = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "model_type": "SARIMAX",
    "num_models": len(models_dict),
    "min_training_months": MIN_MONTHS,
    "prediction_horizon": VALIDATION_MONTHS,
    "groups": sorted(list(models_dict.keys())),
    "features": feature_cols,
    "architecture": {
        "base_order": "(1,1,1)",
        "seasonal_order": "(0,0,0,0)",
        "exogenous_variables": len(feature_cols),
        "trend": "n",
        "note": "SARIMAX with differencing for stability, no linear trend"
    },
    "test_performance": {
        "avg_rmse": float(avg_rmse),
        "avg_mae": float(avg_mae),
        "avg_mape": float(avg_mape)
    },
    "individual_metrics": individual_metrics
}

# Save pickle + gzip pickle + metrics
with open(PKL_PATH, "wb") as f:
    pickle.dump(models_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

with gzip.open(PKL_GZ_PATH, "wb", compresslevel=9) as f:
    pickle.dump(models_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

logger.info(f"Saved models: {PKL_PATH} and {PKL_GZ_PATH}")
logger.info(f"Saved metrics: {METRICS_PATH}")
logger.info(f"Trained {len(models_dict):,} models")
logger.info(f"Avg RMSE: {avg_rmse:.2f}, Avg MAE: {avg_mae:.2f}, Avg MAPE: {avg_mape:.2f}%")
PY
log "✓ Models trained successfully"

log "→ Uploading models + metrics to S3 (gold/)..."
aws s3 cp "$PIPELINE_DIR/sarimax-models-dict.pkl" "s3://$S3_BUCKET/gold/" >> "$LOG_FILE" 2>&1
aws s3 cp "$PIPELINE_DIR/sarimax-models-dict.pkl.gz" "s3://$S3_BUCKET/gold/" >> "$LOG_FILE" 2>&1
aws s3 cp "$PIPELINE_DIR/model-metrics.json" "s3://$S3_BUCKET/gold/" >> "$LOG_FILE" 2>&1
log "✓ Upload completed"

log "→ Generating predictions with updated models..."
python3 "$PIPELINE_DIR/gold_predictions.py" >> "$LOG_FILE" 2>&1
log "✓ Predictions generated successfully"

log "→ Cleanup..."
rm -f "$PIPELINE_DIR/features.parquet" || true
log "✓ Cleanup completed"

find "$LOG_DIR" -name "trimonthly_retrain_*.log" -type f -mtime +90 -delete

log "=========================================="
log "✓ TRIMONTHLY RETRAIN COMPLETED SUCCESSFULLY"
log "=========================================="