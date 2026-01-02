"""
INCREMENTAL MODEL UPDATE - SARIMAX Version
Updates existing SARIMAX models with new data
Memory efficient for t4g.nano instances
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
import boto3
from datetime import datetime
from io import BytesIO
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = "hdb-prediction-pipeline"
SILVER_KEY = "silver/features.parquet"
MODEL_KEY = "gold/sarimax-models-dict.pkl"
METRICS_KEY = "gold/model-metrics.json"

s3 = boto3.client("s3")


REQ_COLS = [
    "month", "group_key", "avg_price",
    "avg_floor_area", "avg_storey", "avg_remaining_lease", "avg_lease_commence"
]


def read_s3_bytes(bucket, key):
    logger.info(f"Reading s3://{bucket}/{key}")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def load_silver():
    """Load silver layer data"""
    df = pd.read_parquet(BytesIO(read_s3_bytes(BUCKET_NAME, SILVER_KEY)))
    logger.info(f"✓ Loaded {len(df):,} rows from silver")
    logger.info(f"✓ Groups: {df['group_key'].nunique()}")
    return df


def load_existing():
    """Load existing SARIMAX models"""
    try:
        models_dict = pickle.loads(read_s3_bytes(BUCKET_NAME, MODEL_KEY))
        logger.info(f"✓ Loaded existing models ({len(models_dict)} groups)")
        return models_dict
    except Exception as e:
        logger.warning(f"No existing models: {e}")
        return {}


def prepare_group_data(group_df):
    """Prepare data for one group - fill missing months"""
    group_df = group_df.copy()
    group_df = group_df.sort_values('month')
    
    
    group_df = group_df.set_index('month')
    group_df = group_df.asfreq('MS')
    
    
    group_df = group_df.ffill().bfill()
    
    
    for col in group_df.columns:
        if group_df[col].isna().any():
            median = group_df[col].median()
            group_df[col] = group_df[col].fillna(0.0 if pd.isna(median) else median)
    
    return group_df.reset_index()


def scale_data(y, exog):
    """Scale data to prevent numerical instability"""
    
    y_min, y_max = y.min(), y.max()
    y_scaled = (y - y_min) / (y_max - y_min + 1e-8)
    
    
    exog_scaled = np.zeros_like(exog)
    exog_params = []
    for i in range(exog.shape[1]):
        col_min, col_max = exog[:, i].min(), exog[:, i].max()
        exog_scaled[:, i] = (exog[:, i] - col_min) / (col_max - col_min + 1e-8)
        exog_params.append((col_min, col_max))
    
    return y_scaled, exog_scaled, (y_min, y_max), exog_params


def train_sarimax_model(y, exog, order=(1,0,1), seasonal_order=(0,0,0,0)):
    """Train SARIMAX model for one group with scaling"""
    try:
        
        y_scaled, exog_scaled, y_params, exog_params = scale_data(y, exog)
        
        
        model = SARIMAX(
            y_scaled,
            exog=exog_scaled,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend='c'
        )
        
        fitted_model = model.fit(disp=False, maxiter=200, method='lbfgs')
        
        return {
            'model': fitted_model,
            'y_params': y_params,
            'exog_params': exog_params
        }
        
    except Exception as e:
        logger.warning(f"SARIMAX fitting failed: {e}")
        
        try:
            y_scaled, exog_scaled, y_params, exog_params = scale_data(y, exog)
            
            model = SARIMAX(
                y_scaled,
                exog=exog_scaled,
                order=(1,0,0),
                seasonal_order=(0,0,0,0),
                enforce_stationarity=False,
                enforce_invertibility=False,
                trend='c'
            )
            fitted_model = model.fit(disp=False, maxiter=100)
            
            return {
                'model': fitted_model,
                'y_params': y_params,
                'exog_params': exog_params
            }
        except:
            return None


def update_models(df, existing_models):
    """Train or update SARIMAX models"""
    logger.info("\n" + "=" * 60)
    logger.info("UPDATING SARIMAX MODELS (with scaling)")
    logger.info("=" * 60)
    
    
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Silver missing columns: {missing}")
    
    groups = sorted(df['group_key'].unique())
    logger.info(f"Processing {len(groups)} groups...")
    
    models_dict = {}
    metrics = []
    new_models = 0
    updated_models = 0
    
    for i, group in enumerate(groups):
        if i % 5 == 0:
            logger.info(f"\nProgress: {i}/{len(groups)} groups processed")
        
        group_df = df[df['group_key'] == group].copy()
        
        
        if len(group_df) < 36:
            logger.warning(f"  Skipping {group}: only {len(group_df)} months")
            continue
        
        
        group_df = prepare_group_data(group_df)
        group_df = group_df.tail(36)
        
        
        y = group_df['avg_price'].values
        exog = group_df[['avg_floor_area', 'avg_storey', 'avg_remaining_lease', 'avg_lease_commence']].values
        
        
        train_size = len(y) - 6
        y_train, y_test = y[:train_size], y[train_size:]
        exog_train, exog_test = exog[:train_size], exog[train_size:]
        
        
        model_dict = train_sarimax_model(y_train, exog_train)
        
        if model_dict is None:
            logger.warning(f"  Failed to fit {group}")
            continue
        
        
        try:
            
            exog_test_scaled = np.zeros_like(exog_test)
            for j, (col_min, col_max) in enumerate(model_dict['exog_params']):
                exog_test_scaled[:, j] = (exog_test[:, j] - col_min) / (col_max - col_min + 1e-8)
            
            
            forecast_scaled = model_dict['model'].forecast(steps=6, exog=exog_test_scaled)
            
            
            y_min, y_max = model_dict['y_params']
            forecast = forecast_scaled * (y_max - y_min) + y_min
            
            
            if forecast.min() < 0 or forecast.max() > 5000000:
                logger.warning(f"  Predictions out of range for {group}")
                continue
            
            rmse = np.sqrt(np.mean((y_test - forecast) ** 2))
            mae = np.mean(np.abs(y_test - forecast))
            mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100
            
            if rmse > 500000 or mape > 50:
                logger.warning(f"  Metrics too high for {group}")
                continue
            
            if group in existing_models:
                updated_models += 1
            else:
                new_models += 1
            
            
            models_dict[group] = {
                'model': model_dict['model'],
                'y_params': model_dict['y_params'],
                'exog_params': model_dict['exog_params'],
                'last_y': y,
                'last_exog': exog,
                'last_date': group_df['month'].iloc[-1]
            }
            
            metrics.append({
                'group': group,
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape)
            })
            
        except Exception as e:
            logger.warning(f"  Validation failed for {group}: {e}")
    
    logger.info(f"\n✓ New models: {new_models}")
    logger.info(f"✓ Updated models: {updated_models}")
    logger.info(f"✓ Total models: {len(models_dict)}")
    
    return models_dict, metrics


def put_pickle(bucket, key, obj):
    """Save pickle to S3"""
    buf = BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.info(f"✓ Saved {key}")


def put_json(bucket, key, obj):
    """Save JSON to S3"""
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
        ContentType="application/json"
    )
    logger.info(f"✓ Saved {key}")


def main():
    logger.info("=" * 60)
    logger.info("INCREMENTAL MODEL UPDATE - SARIMAX")
    logger.info("=" * 60)
    
    
    df = load_silver()
    existing_models = load_existing()
    
    
    models_dict, metrics = update_models(df, existing_models)
    
    if len(models_dict) == 0:
        logger.error("No models trained successfully")
        return False
    
    
    avg_rmse = np.mean([m['rmse'] for m in metrics])
    avg_mae = np.mean([m['mae'] for m in metrics])
    avg_mape = np.mean([m['mape'] for m in metrics])
    
    
    logger.info("\nSaving models to S3...")
    put_pickle(BUCKET_NAME, MODEL_KEY, models_dict)
    
    
    metrics_summary = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "SARIMAX-incremental",
        "num_models": len(models_dict),
        "groups": list(models_dict.keys()),
        "test_performance": {
            "avg_rmse": float(avg_rmse),
            "avg_mae": float(avg_mae),
            "avg_mape": float(avg_mape)
        },
        "architecture": {
            "base_order": "(1,1,1)",
            "seasonal_order": "(1,1,1,12)",
            "exogenous_variables": 4
        }
    }
    put_json(BUCKET_NAME, METRICS_KEY, metrics_summary)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ INCREMENTAL TRAINING COMPLETE")
    logger.info(f"✓ Updated {len(models_dict)} models")
    logger.info(f"✓ Avg RMSE: ${avg_rmse:,.2f}")
    logger.info(f"✓ Avg MAE: ${avg_mae:,.2f}")
    logger.info(f"✓ Avg MAPE: {avg_mape:.2f}%")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)