"""
Gold Layer: Generate Predictions using SARIMAX Models
UPDATED: 7-month minimum with 2-month predictions (next month + month after)
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BUCKET_NAME = 'hdb-prediction-pipeline'
SILVER_KEY = 'silver/features.parquet'
MODEL_KEY = 'gold/sarimax-models-dict.pkl'
PREDICTIONS_KEY = 'gold/predictions-cache.parquet'
DYNAMODB_TABLE = 'hdb-predictions'

# CHANGED: Match training requirement of 7 months
MIN_MONTHS_REQUIRED = 7
# CHANGED: Predict only 2 months ahead
PREDICTION_HORIZON = 2

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)


def get_table_key_schema():
    try:
        table_info = table.key_schema
        logger.info(f"DynamoDB table key schema: {table_info}")
        
        partition_key = None
        sort_key = None
        
        for key in table_info:
            if key['KeyType'] == 'HASH':
                partition_key = key['AttributeName']
            elif key['KeyType'] == 'RANGE':
                sort_key = key['AttributeName']
        
        logger.info(f"Partition key: {partition_key}, Sort key: {sort_key}")
        return partition_key, sort_key
        
    except Exception as e:
        logger.error(f"Failed to get table schema: {e}")
        return 'region_flattype', None


def read_from_s3(bucket, key):
    logger.info(f"Reading s3://{bucket}/{key}")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()


def load_models():
    logger.info("Loading SARIMAX models...")
    models_dict = pickle.loads(read_from_s3(BUCKET_NAME, MODEL_KEY))
    
    if not isinstance(models_dict, dict):
        raise ValueError("Models must be a dict")
    
    logger.info(f"✓ Loaded {len(models_dict)} SARIMAX models")
    logger.info(f"✓ Models for groups: {sorted(models_dict.keys())}")
    return models_dict


def load_silver_data():
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=SILVER_KEY)
    buffer = BytesIO(response['Body'].read())
    df = pd.read_parquet(buffer)
    
    logger.info(f"Initial load from silver: {len(df):,} rows")
    
    if len(df) == 0:
        logger.error("Silver layer is empty!")
        return df
    
    logger.info(f"Columns in silver: {df.columns.tolist()}")
    logger.info(f"Groups in silver: {df['group_key'].nunique() if 'group_key' in df.columns else 'N/A'}")
    
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    
    initial_count = len(df)
    df = df.dropna(subset=['month'])
    if len(df) < initial_count:
        logger.warning(f"Dropped {initial_count - len(df)} rows with invalid dates")
    
    if len(df) == 0:
        logger.error("No valid dates in silver layer!")
        return df
    
    max_date = df['month'].max()
    min_date = df['month'].min()
    logger.info(f"Date range in silver: {min_date} to {max_date}")
    
    logger.info(f"Using all available data (cold start mode)")
    
    logger.info(f"✓ Loaded {len(df):,} rows")
    logger.info(f"✓ Groups: {df['group_key'].nunique()}")
    
    if len(df) == 0:
        logger.error("All data was filtered out!")
    
    return df


def prepare_group_data(group_df):
    """Prepare data for one group - fill missing months"""
    group_df = group_df.copy()
    group_df['month'] = pd.to_datetime(group_df['month'])
    group_df = group_df.sort_values('month')
    
    # Fill missing months
    group_df = group_df.set_index('month').asfreq('MS').reset_index()
    
    # Forward fill then backward fill
    req_cols = ['avg_price', 'avg_floor_area', 'avg_storey', 'avg_remaining_lease', 'avg_lease_commence']
    for col in req_cols:
        if col in group_df.columns:
            group_df[col] = pd.to_numeric(group_df[col], errors='coerce')
            group_df[col] = group_df[col].ffill().bfill()
            
            if group_df[col].isna().any():
                median = group_df[col].median()
                group_df[col] = group_df[col].fillna(0.0 if pd.isna(median) else median)
    
    return group_df


def generate_prediction_for_group(models_dict, group_key, group_df):
    """
    Generate predictions for one group
    UPDATED: 2-month predictions only (next month + month after)
    """
    if group_key not in models_dict:
        logger.warning(f"No model for {group_key}, skipping")
        return None
    
    group_df = prepare_group_data(group_df)
    
    model_data = models_dict[group_key]
    model = model_data.get('model')
    y_params = model_data['y_params']
    exog_params = model_data['exog_params']
    last_date = model_data['last_date']
    
    exog_cols = ['avg_floor_area', 'avg_storey', 'avg_remaining_lease', 'avg_lease_commence']
    
    try:
        # Get current data
        current_y = group_df['avg_price'].values
        current_exog = group_df[exog_cols].values
        
        # CRITICAL: Update model with new data since training
        new_data_mask = group_df['month'] > last_date
        
        if new_data_mask.any():
            logger.info(f"  Updating {group_key} with {new_data_mask.sum()} new observations")
            
            # Get new observations
            new_y = current_y[new_data_mask]
            new_exog = current_exog[new_data_mask]
            
            # Scale new data using training parameters
            y_min, y_max = y_params
            new_y_scaled = (new_y - y_min) / (y_max - y_min + 1e-8)
            
            new_exog_scaled = np.zeros_like(new_exog)
            for j, (col_min, col_max) in enumerate(exog_params):
                new_exog_scaled[:, j] = (new_exog[:, j] - col_min) / (col_max - col_min + 1e-8)
            
            # Update model state with new observations
            model = model.apply(new_y_scaled, exog=new_exog_scaled)
        
        # Prepare exogenous variables for forecast
        # Use last known values for future predictions
        last_exog_values = current_exog[-1]
        exog_forecast = np.tile(last_exog_values, (PREDICTION_HORIZON, 1))
        
        # Scale forecast exogenous variables
        exog_forecast_scaled = np.zeros_like(exog_forecast)
        for j, (col_min, col_max) in enumerate(exog_params):
            exog_forecast_scaled[:, j] = (exog_forecast[:, j] - col_min) / (col_max - col_min + 1e-8)
        
        # Forecast from updated model state
        forecast_scaled = model.forecast(steps=PREDICTION_HORIZON, exog=exog_forecast_scaled)
        
        # Unscale predictions back to original price range
        y_min, y_max = y_params
        forecast = forecast_scaled * (y_max - y_min) + y_min
        
        # Sanity check
        if forecast.min() < 0 or forecast.max() > 5000000:
            logger.warning(f"Predictions out of range for {group_key}: {forecast.min():.0f} to {forecast.max():.0f}")
            return None
        
        # Extract predictions
        current_price = float(current_y[-1])
        
        pred_1m = float(forecast[0])  # Next month
        pred_2m = float(forecast[1])  # Month after next
        
        # Determine trend based on next month prediction
        if pred_1m > current_price * 1.02:
            trend = 'increasing'
        elif pred_1m < current_price * 0.98:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        logger.info(f"  {group_key}: ${current_price:,.0f} → ${pred_1m:,.0f} → ${pred_2m:,.0f} ({trend})")
        
        return {
            'current_avg_price': current_price,
            'predicted_1m_price': pred_1m,
            'predicted_2m_price': pred_2m,
            'trend': trend,
            'confidence_score': 0.80  # Slightly lower for 7-month models
        }
        
    except Exception as e:
        logger.warning(f"Prediction failed for {group_key}: {e}")
        return None


def generate_all_predictions(models_dict, df):
    """Generate predictions for all groups"""
    logger.info("\n" + "=" * 60)
    logger.info(f"Generating {PREDICTION_HORIZON}-month predictions...")
    logger.info("=" * 60)
    
    predictions = []
    skipped = 0
    no_model = 0
    
    group_keys = sorted(df['group_key'].unique())
    
    # Diagnostic: Check overlap
    models_available = set(models_dict.keys())
    groups_in_data = set(group_keys)
    
    logger.info(f"\nDiagnostic Info:")
    logger.info(f"  Models available: {len(models_available)}")
    logger.info(f"  Groups in data: {len(groups_in_data)}")
    logger.info(f"  Overlap: {len(models_available & groups_in_data)}")
    
    missing_models = groups_in_data - models_available
    if missing_models:
        logger.warning(f"  Groups without models: {sorted(missing_models)}")
    
    logger.info("")
    
    for i, group_key in enumerate(group_keys):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(group_keys)} groups processed")
        
        group_df = df[df['group_key'] == group_key].copy()
        
        # Check minimum months requirement
        if len(group_df) < MIN_MONTHS_REQUIRED:
            logger.warning(f"Skipping {group_key}: only {len(group_df)} months (need {MIN_MONTHS_REQUIRED})")
            skipped += 1
            continue
        
        # Check if model exists
        if group_key not in models_dict:
            no_model += 1
            continue
        
        # Generate prediction
        pred = generate_prediction_for_group(models_dict, group_key, group_df)
        
        if pred:
            region = group_df['region'].iloc[0]
            flat_type = group_df['flat_type'].iloc[0]
            
            predictions.append({
                'region': region,
                'flat_type': flat_type,
                'group_key': group_key,
                **pred
            })
        else:
            skipped += 1
        
        del group_df
    
    logger.info(f"\n✓ Generated {len(predictions)} predictions")
    logger.info(f"  Skipped (insufficient data): {skipped}")
    logger.info(f"  Skipped (no model): {no_model}")
    
    return pd.DataFrame(predictions)


def save_predictions_to_s3(df, bucket, key):
    logger.info(f"Saving to s3://{bucket}/{key}")
    
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
    
    logger.info(f"✓ Saved {len(df)} predictions")


def update_dynamodb_cache(predictions_df):
    logger.info("Updating DynamoDB...")
    
    partition_key, sort_key = get_table_key_schema()
    
    success_count = 0
    error_count = 0
    
    with table.batch_writer() as batch:
        for _, row in predictions_df.iterrows():
            try:
                item = {}
                
                if partition_key:
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
                    'predicted_1m_price': Decimal(str(round(row['predicted_1m_price'], 2))),
                    'predicted_2m_price': Decimal(str(round(row['predicted_2m_price'], 2))),
                    'trend': row['trend'],
                    'confidence_score': Decimal(str(row['confidence_score'])),
                    'last_updated': row['last_updated']
                })
                
                batch.put_item(Item=item)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error writing {row['group_key']}: {e}")
                error_count += 1
    
    logger.info(f"✓ DynamoDB: {success_count} success, {error_count} errors")


def main():
    try:
        logger.info("=" * 60)
        logger.info("GOLD LAYER: Prediction Generation")
        logger.info(f"Minimum data required: {MIN_MONTHS_REQUIRED} months")
        logger.info(f"Prediction horizon: {PREDICTION_HORIZON} months (next month + month after)")
        logger.info("=" * 60)
        
        # Load models
        models_dict = load_models()
        
        # Load data
        df = load_silver_data()
        
        # Generate predictions
        predictions_df = generate_all_predictions(models_dict, df)
        
        if len(predictions_df) == 0:
            logger.warning("No predictions generated!")
            return False
        
        # Save to S3
        save_predictions_to_s3(predictions_df, BUCKET_NAME, PREDICTIONS_KEY)
        
        # Update DynamoDB
        update_dynamodb_cache(predictions_df)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total predictions: {len(predictions_df)}")
        logger.info(f"Groups covered: {sorted(predictions_df['group_key'].unique())}")
        logger.info("\nTrend distribution:")
        logger.info(predictions_df['trend'].value_counts().to_string())
        logger.info("\nSample predictions (first 5 groups):")
        for _, row in predictions_df.head(5).iterrows():
            logger.info(f"  {row['group_key']}:")
            logger.info(f"    Current:      ${row['current_avg_price']:,.0f}")
            logger.info(f"    Next month:   ${row['predicted_1m_price']:,.0f}")
            logger.info(f"    Month after:  ${row['predicted_2m_price']:,.0f}")
            logger.info(f"    Trend: {row['trend']}")
        logger.info("=" * 60)
        logger.info("✓ Gold layer completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Gold layer failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)