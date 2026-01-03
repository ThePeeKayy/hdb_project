"""
Gold Predictions - NON-STATIONARY SARIMAX
Configured for non-stationary time series
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import pickle
import logging
from datetime import datetime
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BUCKET_NAME = 'hdb-prediction-pipeline'
SILVER_KEY = 'silver/features.parquet'
MODEL_KEY = 'gold/sarimax-models-dict.pkl'
PREDICTIONS_KEY = 'gold/predictions-cache.parquet'
DYNAMODB_TABLE = 'hdb-predictions'

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)


def get_table_schema():
    """Get actual DynamoDB key schema"""
    try:
        response = table.key_schema
        partition_key = None
        sort_key = None
        
        for key in response:
            if key['KeyType'] == 'HASH':
                partition_key = key['AttributeName']
            elif key['KeyType'] == 'RANGE':
                sort_key = key['AttributeName']
        
        logger.info(f"DynamoDB schema: PK={partition_key}, SK={sort_key}")
        return partition_key, sort_key
    except Exception as e:
        logger.error(f"Failed to get schema: {e}")
        
        return 'region_flattype', None


def load_models():
    import gzip
    logger.info("Loading models...")
    
    
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY + '.gz')
        with gzip.GzipFile(fileobj=BytesIO(response['Body'].read())) as f:
            models_dict = pickle.load(f)
        logger.info(f"✓ Loaded {len(models_dict)} models (compressed)")
    except:
        
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
        models_dict = pickle.loads(response['Body'].read())
        logger.info(f"✓ Loaded {len(models_dict)} models")
    
    return models_dict


def load_silver_data():
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=SILVER_KEY)
    df = pd.read_parquet(BytesIO(response['Body'].read()))
    
    df['month'] = pd.to_datetime(df['month'])
    df = df.dropna(subset=['month'])
    
    logger.info(f"✓ Loaded {len(df):,} rows")
    logger.info(f"  Date range: {df['month'].min()} to {df['month'].max()}")
    return df


def predict_for_group(models_dict, group_key, group_df):
    """Prediction using non-stationary SARIMAX models"""
    if group_key not in models_dict:
        return None
    
    model_data = models_dict[group_key]
    model = model_data['model']
    last_date = model_data['last_date']
    
    try:
        group_df = group_df.sort_values('month').copy()
        
        ts = group_df.set_index('month')['avg_price']
        ts = ts.asfreq('MS')
        ts = ts.interpolate(method='time')
        
        
        forecast = model.forecast(steps=12)
        
        current_price = float(ts.iloc[-1])
        pred_6 = float(forecast.iloc[5])
        pred_12 = float(forecast.iloc[11])
        
        change_6m = (pred_6 - current_price) / current_price
        
        if change_6m > 0.01:
            trend = 'increasing'
        elif change_6m < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        if abs(change_6m) > 0.10:
            confidence = 0.60
        else:
            confidence = 0.85
        
        return {
            'current_avg_price': current_price,
            'predicted_6m_price': pred_6,
            'predicted_12m_price': pred_12,
            'trend': trend,
            'confidence_score': confidence,
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"{group_key}: {e}")
        return None


def generate_all(models_dict, df):
    logger.info("\nGenerating predictions...")
    
    predictions = []
    
    for group_key in sorted(df['group_key'].unique()):
        group_df = df[df['group_key'] == group_key].copy()
        
        pred = predict_for_group(models_dict, group_key, group_df)
        
        if pred:
            region = group_df['region'].iloc[0]
            flat_type = group_df['flat_type'].iloc[0]
            
            predictions.append({
                'region': region,
                'flat_type': flat_type,
                'group_key': group_key,
                **pred
            })
    
    logger.info(f"✓ Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)


def save_to_s3(df):
    buffer = BytesIO()
    df.to_parquet(buffer, compression='snappy', index=False)
    buffer.seek(0)
    
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=PREDICTIONS_KEY,
        Body=buffer.getvalue()
    )
    logger.info("✓ Saved to S3")


def update_dynamodb(predictions_df):
    """Update DynamoDB with correct schema"""
    logger.info("Updating DynamoDB...")
    
    partition_key, sort_key = get_table_schema()
    
    success = 0
    errors = 0
    
    with table.batch_writer() as batch:
        for _, row in predictions_df.iterrows():
            try:
                
                item = {}
                
                
                if partition_key == 'region_flattype':
                    item[partition_key] = row['group_key']
                elif partition_key == 'group_key':
                    item[partition_key] = row['group_key']
                elif partition_key == 'region':
                    item[partition_key] = row['region']
                else:
                    item[partition_key] = row['group_key']
                
                
                if sort_key:
                    if sort_key == 'timestamp' or sort_key == 'last_updated':
                        item[sort_key] = row['last_updated']
                    elif sort_key == 'region':
                        item[sort_key] = row['region']
                    elif sort_key == 'flat_type':
                        item[sort_key] = row['flat_type']
                    else:
                        item[sort_key] = row['last_updated']
                
                
                item.update({
                    'region': row['region'],
                    'flat_type': row['flat_type'],
                    'group_key': row['group_key'],
                    'current_avg_price': Decimal(str(round(row['current_avg_price'], 2))),
                    'predicted_6m_price': Decimal(str(round(row['predicted_6m_price'], 2))),
                    'predicted_12m_price': Decimal(str(round(row['predicted_12m_price'], 2))),
                    'trend': row['trend'],
                    'confidence_score': Decimal(str(row['confidence_score'])),
                    'last_updated': row['last_updated']
                })
                
                batch.put_item(Item=item)
                success += 1
                
            except Exception as e:
                logger.error(f"Error on {row['group_key']}: {e}")
                errors += 1
    
    logger.info(f"✓ DynamoDB: {success} success, {errors} errors")


def main():
    logger.info("="*60)
    logger.info("PREDICTIONS - NON-STATIONARY SARIMAX")
    logger.info("="*60)
    
    models_dict = load_models()
    df = load_silver_data()
    
    predictions_df = generate_all(models_dict, df)
    
    if len(predictions_df) == 0:
        logger.error("No predictions!")
        return False
    
    save_to_s3(predictions_df)
    update_dynamodb(predictions_df)
    
    
    logger.info("\n" + "="*60)
    logger.info(f"Total: {len(predictions_df)}")
    
    predictions_df['change_pct'] = (predictions_df['predicted_6m_price'] - predictions_df['current_avg_price']) / predictions_df['current_avg_price'] * 100
    logger.info(f"Avg change: {predictions_df['change_pct'].mean():.2f}%")
    logger.info(f"Range: {predictions_df['change_pct'].min():.2f}% to {predictions_df['change_pct'].max():.2f}%")
    
    logger.info("\nSamples:")
    for _, row in predictions_df.head(5).iterrows():
        change = (row['predicted_6m_price'] - row['current_avg_price']) / row['current_avg_price'] * 100
        logger.info(f"  {row['group_key']}: ${row['current_avg_price']:,.0f} → ${row['predicted_6m_price']:,.0f} ({change:+.1f}%)")
    
    logger.info("="*60)
    logger.info("✓ DONE")
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)