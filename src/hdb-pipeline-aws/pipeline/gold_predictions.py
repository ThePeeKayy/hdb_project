"""
Gold Layer: Generate Predictions using Trained TiDE Model
Runs daily on EC2 t4g.micro
Execution time: ~5 minutes
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import pickle
import logging
from datetime import datetime
from decimal import Decimal


from darts import TimeSeries


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


BUCKET_NAME = 'hdb-prediction-pipeline'
SILVER_KEY = 'silver/features.parquet'
MODEL_KEY = 'gold/tide-model.pkl'
SCALER_KEY = 'gold/tide-scaler.pkl'
PREDICTIONS_KEY = 'gold/predictions-cache.parquet'
DYNAMODB_TABLE = 'hdb-predictions'


s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)


def read_from_s3(bucket, key):
    
    logger.info(f"Reading from s3://{bucket}/{key}")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()


def load_model_and_scalers():
    
    logger.info("Loading model and scalers...")
    
    
    model_bytes = read_from_s3(BUCKET_NAME, MODEL_KEY)
    model = pickle.loads(model_bytes)
    
    
    scaler_bytes = read_from_s3(BUCKET_NAME, SCALER_KEY)
    scalers = pickle.loads(scaler_bytes)
    
    logger.info("Model and scalers loaded successfully")
    
    return model, scalers['target_scaler'], scalers['covariate_scaler']


def load_silver_data():
    
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=SILVER_KEY)
    buffer = BytesIO(response['Body'].read())
    df = pd.read_parquet(buffer)
    
    logger.info(f"Loaded {len(df)} feature rows")
    return df


def get_latest_data_by_group(df):
    
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values('month')
    
    
    group_data = {}
    
    for group_key in df['group_key'].unique():
        group_df = df[df['group_key'] == group_key].tail(12)
        
        if len(group_df) >= 12:  
            group_data[group_key] = group_df
    
    logger.info(f"Found {len(group_data)} groups with sufficient data")
    
    return group_data


def generate_predictions_for_group(model, target_scaler, covariate_scaler, group_df, feature_cols):
    try:
        
        target_series = TimeSeries.from_dataframe(
            group_df,
            time_col='month',
            value_cols='avg_price',
            freq='MS'
        )
        
        
        covariate_cols = [col for col in feature_cols if col != 'avg_price']
        covariate_series = TimeSeries.from_dataframe(
            group_df,
            time_col='month',
            value_cols=covariate_cols,
            freq='MS'
        )
        
        
        target_scaled = target_scaler.transform(target_series)
        covariates_scaled = covariate_scaler.transform(covariate_series)
        
        
        prediction_scaled = model.predict(
            n=12,
            series=target_scaled,
            past_covariates=covariates_scaled
        )
        
        
        prediction = target_scaler.inverse_transform(prediction_scaled)
        
        
        pred_values = prediction.values().flatten()
        
        
        current_avg = group_df['avg_price'].iloc[-1]
        
        
        pred_6m = float(np.mean(pred_values[:6]))
        
        
        pred_12m = float(np.mean(pred_values[6:12]))
        
        
        if pred_12m > current_avg * 1.02:
            trend = 'increasing'
        elif pred_12m < current_avg * 0.98:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'current_avg_price': float(current_avg),
            'predicted_6m_price': pred_6m,
            'predicted_12m_price': pred_12m,
            'trend': trend,
            'confidence_score': 0.85  
        }
        
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return None


def generate_all_predictions(model, target_scaler, covariate_scaler, group_data, feature_cols):
    
    predictions = []
    
    for group_key, group_df in group_data.items():
        pred = generate_predictions_for_group(
            model, target_scaler, covariate_scaler, group_df, feature_cols
        )
        
        if pred:
            
            parts = group_key.split('_')
            
            
            if len(parts) >= 3:
                town = '_'.join(parts[:-2])
                flat_type = parts[-2]
                lease_bucket = parts[-1]
            else:
                logger.warning(f"Unexpected group_key format: {group_key}")
                continue
            
            predictions.append({
                'town': town,
                'flat_type': flat_type,
                'remaining_lease_bucket': lease_bucket,
                'group_key': group_key,
                **pred
            })
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    return pd.DataFrame(predictions)


def save_predictions_to_s3(df, bucket, key):
    logger.info(f"Saving predictions to s3://{bucket}/{key}")
    
    df['last_updated'] = datetime.now().isoformat()
    
    buffer = BytesIO()
    df.to_parquet(buffer, compression='snappy', index=False)
    
    buffer.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType='application/x-parquet'
    )
    
    logger.info(f"Saved {len(df)} predictions to S3")


def update_dynamodb_cache(predictions_df):
    
    success_count = 0
    error_count = 0
    
    with table.batch_writer() as batch:
        for _, row in predictions_df.iterrows():
            try:
                item = {
                    'town_flattype_lease': row['group_key'],
                    'town': row['town'],
                    'flat_type': row['flat_type'],
                    'remaining_lease_bucket': row['remaining_lease_bucket'],
                    'current_avg_price': Decimal(str(row['current_avg_price'])),
                    'predicted_6m_price': Decimal(str(row['predicted_6m_price'])),
                    'predicted_12m_price': Decimal(str(row['predicted_12m_price'])),
                    'trend': row['trend'],
                    'confidence_score': Decimal(str(row['confidence_score'])),
                    'last_updated': row['last_updated']
                }
                
                batch.put_item(Item=item)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error writing to DynamoDB: {e}")
                error_count += 1
    
    logger.info(f"DynamoDB update complete: {success_count} success, {error_count} errors")


def main():
    try:
        logger.info("=" * 60)
        logger.info("Starting Prediction Generation Pipeline")
        logger.info("=" * 60)
        
        
        model, target_scaler, covariate_scaler = load_model_and_scalers()
        
        
        df = load_silver_data()
        
        group_data = get_latest_data_by_group(df)
        
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
        
        predictions_df = generate_all_predictions(
            model, target_scaler, covariate_scaler, group_data, feature_cols
        )
        
        if len(predictions_df) == 0:
            logger.warning("No predictions generated")
            return False
        
        save_predictions_to_s3(predictions_df, BUCKET_NAME, PREDICTIONS_KEY)
        
        update_dynamodb_cache(predictions_df)
        
        logger.info("=" * 60)
        logger.info("Prediction Generation Completed Successfully!")
        logger.info(f"Total Predictions: {len(predictions_df)}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)