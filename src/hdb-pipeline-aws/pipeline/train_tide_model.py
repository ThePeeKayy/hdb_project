"""
Gold Layer: TiDE Model Training (Memory Optimized for t4g.micro)
Runs weekly on EC2 t4g.micro
Execution time: ~30-40 minutes
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import pickle
import json
import logging
from datetime import datetime, timedelta
from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import rmse, mae, mape
from darts.dataprocessing.transformers import Scaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


BUCKET_NAME = 'hdb-prediction-pipeline'
SILVER_KEY = 'silver/features.parquet'
MODEL_KEY = 'gold/tide-model.pkl'
METRICS_KEY = 'metrics/model-performance.json'
SCALER_KEY = 'gold/tide-scaler.pkl'


MODEL_CONFIG = {
    'input_chunk_length': 12,       
    'output_chunk_length': 12,      
    'num_encoder_layers': 1,        
    'num_decoder_layers': 1,
    'decoder_output_dim': 4,
    'hidden_size': 64,              
    'temporal_width_past': 4,
    'temporal_decoder_hidden': 32,
    'use_layer_norm': True,
    'dropout': 0.1,
    'batch_size': 16,               
    'n_epochs': 50,                 
    'random_state': 42,
    'force_reset': True,
    'save_checkpoints': False,      
    'pl_trainer_kwargs': {
        'accelerator': 'cpu',
        'enable_progress_bar': False,
        'enable_model_summary': False,
    }
}


s3_client = boto3.client('s3')


def read_from_s3(bucket, key):
    """Read Parquet file from S3"""
    logger.info(f"Reading from s3://{bucket}/{key}")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    buffer = BytesIO(response['Body'].read())
    df = pd.read_parquet(buffer)
    logger.info(f"Loaded {df.shape[0]} rows")
    return df


def prepare_time_series_data(df):
    """
    Prepare data for TiDE model
    Create separate time series for each group
    """
    logger.info("Preparing time series data...")
    
    
    df['month'] = pd.to_datetime(df['month'])
    
    
    df = df.sort_values(['group_key', 'month'])
    
    feature_cols = [
        'avg_price',  
        'avg_price_lag_1m',
        'avg_price_lag_3m',
        'avg_price_roll_3m',
        'avg_price_roll_6m',
        'price_change_1m',
        'price_change_3m',
        'month_sin',
        'month_cos',
        'price_per_sqm',
        'lease_midpoint',
        'flat_type_encoded',
        'price_vs_market',
        'transaction_count'
    ]
    
    df_clean = df.dropna(subset=feature_cols)
    
    logger.info(f"After removing NaN: {len(df_clean)} rows")
    
    return df_clean, feature_cols


def create_train_test_split(df, group_key, test_months=6):
    group_df = df[df['group_key'] == group_key].sort_values('month')
    
    if len(group_df) < 24:  
        return None, None
    
    
    split_idx = len(group_df) - test_months
    
    train_df = group_df.iloc[:split_idx]
    test_df = group_df.iloc[split_idx:]
    
    return train_df, test_df


def train_model_for_group(train_df, test_df, feature_cols):
    try:
        
        train_target = TimeSeries.from_dataframe(
            train_df,
            time_col='month',
            value_cols='avg_price',
            freq='MS'  
        )
        
        test_target = TimeSeries.from_dataframe(
            test_df,
            time_col='month',
            value_cols='avg_price',
            freq='MS'
        )
        
        
        covariate_cols = [col for col in feature_cols if col != 'avg_price']
        
        train_covariates = TimeSeries.from_dataframe(
            train_df,
            time_col='month',
            value_cols=covariate_cols,
            freq='MS'
        )
        
        test_covariates = TimeSeries.from_dataframe(
            test_df,
            time_col='month',
            value_cols=covariate_cols,
            freq='MS'
        )
        
        
        target_scaler = Scaler()
        covariate_scaler = Scaler()
        
        train_target_scaled = target_scaler.fit_transform(train_target)
        train_covariates_scaled = covariate_scaler.fit_transform(train_covariates)
        
        test_covariates_scaled = covariate_scaler.transform(test_covariates)
        
        
        model = TiDEModel(**MODEL_CONFIG)
        
        
        model.fit(
            series=train_target_scaled,
            past_covariates=train_covariates_scaled,
            verbose=False
        )
        
        
        prediction_scaled = model.predict(
            n=len(test_target),
            past_covariates=test_covariates_scaled,
            series=train_target_scaled
        )
        
        
        prediction = target_scaler.inverse_transform(prediction_scaled)
        
        
        rmse_score = rmse(test_target, prediction)
        mae_score = mae(test_target, prediction)
        mape_score = mape(test_target, prediction)
        
        return {
            'model': model,
            'target_scaler': target_scaler,
            'covariate_scaler': covariate_scaler,
            'rmse': rmse_score,
            'mae': mae_score,
            'mape': mape_score,
            'train_size': len(train_df),
            'test_size': len(test_df)
        }
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None


def train_global_model(df, feature_cols, sample_groups=20):
    """
    Train a single global model on sampled groups (to fit in memory)
    Then evaluate on holdout test set
    """
    logger.info(f"Training global model on {sample_groups} sampled groups...")
    
    
    unique_groups = df['group_key'].unique()
    
    
    group_sizes = df.groupby('group_key').size()
    top_groups = group_sizes.nlargest(sample_groups).index.tolist()
    
    logger.info(f"Selected top {len(top_groups)} groups by data volume")
    
    train_list = []
    test_list = []
    
    for group in top_groups:
        train_df, test_df = create_train_test_split(df, group, test_months=6)
        
        if train_df is not None and test_df is not None:
            train_list.append(train_df)
            test_list.append(test_df)
    
    if not train_list:
        logger.error("No valid groups for training")
        return None
    
    train_combined = pd.concat(train_list, ignore_index=True)
    test_combined = pd.concat(test_list, ignore_index=True)
    
    logger.info(f"Training on {len(train_combined)} observations")
    logger.info(f"Testing on {len(test_combined)} observations")
    
    result = train_model_for_group(train_combined, test_combined, feature_cols)
    
    if result:
        logger.info(f"Model trained successfully!")
        logger.info(f"RMSE: ${result['rmse']:,.2f}")
        logger.info(f"MAE: ${result['mae']:,.2f}")
        logger.info(f"MAPE: {result['mape']:.2%}")
    
    return result


def save_model_to_s3(model_data, bucket):
    logger.info("Saving model to S3...")
    
    model_buffer = BytesIO()
    pickle.dump(model_data['model'], model_buffer)
    model_buffer.seek(0)
    
    s3_client.put_object(
        Bucket=bucket,
        Key=MODEL_KEY,
        Body=model_buffer.getvalue()
    )
    
    scaler_buffer = BytesIO()
    pickle.dump({
        'target_scaler': model_data['target_scaler'],
        'covariate_scaler': model_data['covariate_scaler']
    }, scaler_buffer)
    scaler_buffer.seek(0)
    
    s3_client.put_object(
        Bucket=bucket,
        Key=SCALER_KEY,
        Body=scaler_buffer.getvalue()
    )
    
    logger.info("Model and scalers saved successfully")


def save_metrics_to_s3(metrics, bucket):
    logger.info("Saving metrics to S3...")
    
    metrics_json = json.dumps(metrics, indent=2, default=str)
    
    s3_client.put_object(
        Bucket=bucket,
        Key=METRICS_KEY,
        Body=metrics_json.encode('utf-8'),
        ContentType='application/json'
    )
    
    logger.info("Metrics saved successfully")


def main():
    try:
        
        df = read_from_s3(BUCKET_NAME, SILVER_KEY)
        
        df_clean, feature_cols = prepare_time_series_data(df)
        
        result = train_global_model(df_clean, feature_cols, sample_groups=20)
        
        if result is None:
            logger.error("Model training failed")
            return False
        
        save_model_to_s3(result, BUCKET_NAME)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'TiDE',
            'rmse': float(result['rmse']),
            'mae': float(result['mae']),
            'mape': float(result['mape']),
            'train_size': result['train_size'],
            'test_size': result['test_size'],
            'config': MODEL_CONFIG
        }
        
        save_metrics_to_s3(metrics, BUCKET_NAME)
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)