"""
SARIMAX QUARTERLY TRAINING - Automated S3-based training
Trains fresh SARIMAX models every 3 months using Silver layer features from S3
Replaces existing models completely with new training

SCHEDULED: Run by cron job every 3 months
INPUT: s3://hdb-prediction-pipeline/silver/features.parquet
OUTPUT: 
    - s3://hdb-prediction-pipeline/gold/sarimax-models-dict.pkl
    - s3://hdb-prediction-pipeline/gold/model-metrics.json
"""

import json
import pickle
import logging
from datetime import datetime
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


BUCKET_NAME = 'hdb-prediction-pipeline'
SILVER_KEY = 'silver/features.parquet'
MODEL_KEY = 'gold/sarimax-models-dict.pkl'
METRICS_KEY = 'gold/model-metrics.json'


MIN_MONTHS = 50
VALIDATION_MONTHS = 5
MAX_MAPE = 20.0


s3_client = boto3.client('s3')


def load_silver_data():
    """Load features.parquet from S3 Silver layer"""
    logger.info(f"Loading data from s3://{BUCKET_NAME}/{SILVER_KEY}")
    
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=SILVER_KEY)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        df['month'] = pd.to_datetime(df['month'])
        df = df.dropna(subset=['month'])
        
        logger.info(f"✓ Loaded {len(df):,} rows")
        logger.info(f"  Date range: {df['month'].min()} to {df['month'].max()}")
        logger.info(f"  Groups: {df['group_key'].nunique() if 'group_key' in df.columns else 'N/A'}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data from S3: {e}")
        raise


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare silver data for training"""
    
    
    required = {"month", "region", "flat_type", "avg_price"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"features.parquet is missing required columns: {missing}")
    
    
    if "group_key" not in df.columns:
        df["group_key"] = df["region"].astype(str) + "_" + df["flat_type"].astype(str)
    
    
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    
    
    df = df.dropna(subset=["month", "group_key", "avg_price"])
    df = df[df["avg_price"] > 0]
    
    logger.info(f"✓ Validated data: {len(df):,} rows, {df['group_key'].nunique()} groups")
    
    return df


def train_sarimax_for_group(group_df: pd.DataFrame):
    """
    Train SARIMAX model for one group
    - Conservative approach: (1,1,1) with no trend
    - Returns fitted model and validation metrics
    """
    group_df = group_df.sort_values("month").copy()
    
    ts = group_df.set_index("month")["avg_price"]
    ts = ts.asfreq("MS")  
    ts = ts.interpolate(method="time")  
    
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
        
        
        errors = ts_val - fc
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / ts_val)) * 100
        
        
        if mape > MAX_MAPE:
            logger.warning(f"  MAPE too high: {mape:.2f}% (threshold: {MAX_MAPE}%)")
            return None, {"rmse": rmse, "mae": mae, "mape": mape, "training_months": train_size}
        
        
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
        logger.warning(f"  Fit failed: {e}")
        return None, None


def train_all_models(df: pd.DataFrame):
    """Train SARIMAX models for all groups"""
    
    logger.info("\n" + "=" * 60)
    logger.info("SARIMAX QUARTERLY TRAINING")
    logger.info("Training fresh models from scratch")
    logger.info("order=(1,1,1), trend='n', seasonal=(0,0,0,0)")
    logger.info("=" * 60)
    
    groups = sorted(df["group_key"].unique())
    logger.info(f"\nProcessing {len(groups)} groups...")
    
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
        
        
        models_dict[group] = {
            "model": model,
            "last_date": last_date
        }
        
        individual_metrics.append({
            "group": group,
            "rmse": float(metrics_dict["rmse"]),
            "mae": float(metrics_dict["mae"]),
            "mape": float(metrics_dict["mape"]),
            "training_months": int(metrics_dict["training_months"])
        })
    
    logger.info(f"\n✓ Successfully trained {len(models_dict)} models")
    
    return models_dict, individual_metrics


def create_metrics_json(models_dict, individual_metrics, df):
    """Create comprehensive metrics JSON matching expected format"""
    
    
    avg_rmse = np.mean([m["rmse"] for m in individual_metrics])
    avg_mae = np.mean([m["mae"] for m in individual_metrics])
    avg_mape = np.mean([m["mape"] for m in individual_metrics])
    
    
    feature_cols = [
        col for col in df.columns 
        if col not in ["month", "region", "flat_type", "avg_price", "group_key"]
    ]
    
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
            "note": "SARIMAX with differencing, quarterly full retraining"
        },
        "test_performance": {
            "avg_rmse": float(avg_rmse),
            "avg_mae": float(avg_mae),
            "avg_mape": float(avg_mape)
        },
        "individual_metrics": individual_metrics
    }
    
    return metrics_output


def upload_to_s3(models_dict, metrics_json):
    """Upload trained models and metrics to S3 Gold layer"""
    
    logger.info("\nUploading results to S3...")
    
    try:
        
        model_buffer = BytesIO()
        pickle.dump(models_dict, model_buffer, protocol=pickle.HIGHEST_PROTOCOL)
        model_buffer.seek(0)
        
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=MODEL_KEY,
            Body=model_buffer.getvalue(),
            ContentType='application/octet-stream'
        )
        logger.info(f"✓ Models uploaded to s3://{BUCKET_NAME}/{MODEL_KEY}")
        
        
        metrics_buffer = BytesIO(json.dumps(metrics_json, indent=2).encode('utf-8'))
        
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=METRICS_KEY,
            Body=metrics_buffer.getvalue(),
            ContentType='application/json'
        )
        logger.info(f"✓ Metrics uploaded to s3://{BUCKET_NAME}/{METRICS_KEY}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise


def main():
    """Main training pipeline"""
    
    try:
        logger.info("\n" + "=" * 60)
        logger.info("STARTING QUARTERLY SARIMAX TRAINING")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        
        df = load_silver_data()
        df = validate_data(df)
        
        
        models_dict, individual_metrics = train_all_models(df)
        
        if len(models_dict) == 0:
            logger.error("No models trained!")
            return False
        
        
        metrics_json = create_metrics_json(models_dict, individual_metrics, df)
        
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info(f"  Total models trained: {len(models_dict)}")
        logger.info(f"  Average RMSE: ${metrics_json['test_performance']['avg_rmse']:,.2f}")
        logger.info(f"  Average MAE: ${metrics_json['test_performance']['avg_mae']:,.2f}")
        logger.info(f"  Average MAPE: {metrics_json['test_performance']['avg_mape']:.2f}%")
        logger.info("=" * 60)
        
        
        upload_to_s3(models_dict, metrics_json)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ QUARTERLY TRAINING COMPLETE!")
        logger.info(f"✓ {len(models_dict)} models successfully trained and uploaded")
        logger.info("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)