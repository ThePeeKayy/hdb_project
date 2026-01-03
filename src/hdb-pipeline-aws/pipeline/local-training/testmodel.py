"""
SARIMAX Training - Train directly on Silver features.parquet

Purpose:
- Ensures training is perfectly aligned with Silver + Gold.
- No re-engineering from CSV here; assumes features.parquet is produced by Silver.

Input:
- features.parquet (same schema as your silver layer output)

Output:
- sarimax-models-dict.pkl (+ .gz)
- model-metrics.json

Modeling:
- SARIMAX(order=(1,1,1), seasonal=(0,0,0,0), trend='n')
  Stable for monthly price levels with slowing growth (avoids runaway linear level trend).
"""

import json
import gzip
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Files
SILVER_FEATURES_PATH = Path("features.parquet")  # put your silver output here
MODEL_PATH = Path("sarimax-models-dict.pkl")
METRICS_PATH = Path("model-metrics.json")

# Training gates
MIN_MONTHS = 50
VALIDATION_MONTHS = 5
MAX_MAPE = 20.0


def load_silver_features() -> pd.DataFrame:
    if not SILVER_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SILVER_FEATURES_PATH}. Put your silver output parquet in the same folder."
        )

    logger.info(f"Loading silver features from {SILVER_FEATURES_PATH} ...")
    df = pd.read_parquet(SILVER_FEATURES_PATH)

    # Required cols for this trainer
    required = {"month", "region", "flat_type", "avg_price"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"features.parquet is missing required columns: {missing}")

    # Ensure group_key exists (silver usually provides it)
    if "group_key" not in df.columns:
        df["group_key"] = df["region"].astype(str) + "_" + df["flat_type"].astype(str)

    # Types
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")

    # Basic sanity filtering
    df = df.dropna(subset=["month", "group_key", "avg_price"])
    df = df[df["avg_price"] > 0]

    logger.info(f"✓ Loaded {len(df):,} rows")
    logger.info(f"  Date range: {df['month'].min()} to {df['month'].max()}")
    logger.info(f"  Groups: {df['group_key'].nunique()}")

    return df


def train_sarimax_for_group(group_df: pd.DataFrame):
    """
    Conservative SARIMAX for monthly level series.
    - d=1: models changes (helps with "increasing at a decreasing rate")
    - trend='n': no explicit linear trend in levels (prevents runaway extrapolation)
    """
    group_df = group_df.sort_values("month").copy()

    ts = group_df.set_index("month")["avg_price"]
    ts = ts.asfreq("MS")  # monthly
    ts = ts.interpolate(method="time")  # fill gaps

    if len(ts) < MIN_MONTHS + VALIDATION_MONTHS:
        return None, None

    train_size = len(ts) - VALIDATION_MONTHS
    ts_train = ts.iloc[:train_size]
    ts_val = ts.iloc[train_size:]

    try:
        model = SARIMAX(
            ts_train,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fitted = model.fit(disp=False, maxiter=200)

        fc = fitted.forecast(steps=len(ts_val))
        
        # Calculate all metrics
        errors = ts_val - fc
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / ts_val)) * 100

        if mape > MAX_MAPE:
            return None, {"rmse": rmse, "mae": mae, "mape": mape, "training_months": train_size}

        # Refit on full series for production model
        model_full = SARIMAX(
            ts,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fitted_full = model_full.fit(disp=False, maxiter=200)

        metrics_dict = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "training_months": train_size
        }

        return fitted_full, metrics_dict

    except Exception as e:
        logger.warning(f"Fit failed: {e}")
        return None, None


def train_all(df: pd.DataFrame):
    logger.info("\n" + "=" * 60)
    logger.info("SARIMAX TRAINING (from silver features.parquet)")
    logger.info("order=(1,1,1), trend='n', seasonal=(0,0,0,0)")
    logger.info("=" * 60)

    groups = sorted(df["group_key"].unique())

    models_dict = {}
    individual_metrics = []

    for i, group in enumerate(groups):
        logger.info(f"\n[{i+1}/{len(groups)}] {group}")

        group_df = df[df["group_key"] == group].copy()

        model, metrics_dict = train_sarimax_for_group(group_df)
        if model is None:
            logger.info("  ✗ Failed/rejected")
            continue

        logger.info(f"  ✓ RMSE: {metrics_dict['rmse']:.2f}, MAE: {metrics_dict['mae']:.2f}, MAPE: {metrics_dict['mape']:.2f}%")

        last_date = group_df["month"].max()

        # Gold expects this structure
        models_dict[group] = {"model": model, "last_date": last_date}

        individual_metrics.append({
            "group": group,
            "rmse": float(metrics_dict["rmse"]),
            "mae": float(metrics_dict["mae"]),
            "mape": float(metrics_dict["mape"]),
            "training_months": int(metrics_dict["training_months"])
        })

    logger.info(f"\n✓ Trained {len(models_dict)} models")
    return models_dict, individual_metrics


def save_models(models_dict, individual_metrics, df):
    # Calculate average metrics
    avg_rmse = np.mean([m["rmse"] for m in individual_metrics])
    avg_mae = np.mean([m["mae"] for m in individual_metrics])
    avg_mape = np.mean([m["mape"] for m in individual_metrics])
    
    # Get feature columns (exclude core columns)
    feature_cols = [col for col in df.columns if col not in ["month", "region", "flat_type", "avg_price", "group_key"]]
    
    # Build comprehensive metrics JSON
    metrics_output = {
        "timestamp": datetime.now().isoformat(),
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

    # Save pickled models
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(str(MODEL_PATH) + ".gz", "wb", compresslevel=9) as f:
        pickle.dump(models_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=2)

    size_pkl = MODEL_PATH.stat().st_size / (1024 * 1024)
    size_gz = Path(str(MODEL_PATH) + ".gz").stat().st_size / (1024 * 1024)

    logger.info(f"✓ Saved to {MODEL_PATH}")
    logger.info(f"  Uncompressed: {size_pkl:.1f} MB")
    logger.info(f"  Compressed (.gz): {size_gz:.1f} MB")
    logger.info(f"\n✓ Metrics saved to {METRICS_PATH}")
    logger.info(f"  Avg RMSE: {avg_rmse:.2f}")
    logger.info(f"  Avg MAE: {avg_mae:.2f}")
    logger.info(f"  Avg MAPE: {avg_mape:.2f}%")


def main():
    df = load_silver_features()
    models_dict, individual_metrics = train_all(df)

    if not models_dict:
        logger.error("No models trained!")
        return False

    save_models(models_dict, individual_metrics, df)
    logger.info("\n✓ DONE")
    return True


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)