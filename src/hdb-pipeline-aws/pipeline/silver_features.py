"""
Silver Layer: Time Series Feature Engineering
Runs on EC2 t4g.micro
Execution time: ~3 minutes
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import pyarrow.parquet as pq
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BUCKET_NAME = 'hdb-prediction-pipeline'
BRONZE_KEY = 'bronze/resale-data.parquet'
SILVER_KEY = 'silver/features.parquet'

s3_client = boto3.client('s3')


def read_from_s3(bucket, key):
    """
    Read Parquet file from S3 into DataFrame
    """
    logger.info(f"Reading from s3://{bucket}/{key}")
    
    response = s3_client.get_object(Bucket=bucket, Key=key)
    buffer = BytesIO(response['Body'].read())
    
    df = pd.read_parquet(buffer)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def create_time_series_groups(df):    
    df['month'] = pd.to_datetime(df['month'])
    grouped = df.groupby([
        'month', 'town', 'flat_type', 'remaining_lease_bucket'
    ]).agg({
        'resale_price': ['mean', 'median', 'std', 'count'],
        'floor_area_sqm': 'mean'
    }).reset_index()
    
    grouped.columns = [
        'month', 'town', 'flat_type', 'remaining_lease_bucket',
        'avg_price', 'median_price', 'price_std', 'transaction_count',
        'avg_floor_area'
    ]
    
    grouped['price_per_sqm'] = grouped['avg_price'] / grouped['avg_floor_area']
    grouped = grouped.sort_values(['town', 'flat_type', 'remaining_lease_bucket', 'month'])
        
    return grouped


def add_lag_features(df, periods=[1, 3, 6, 12]):
    
    df['group_key'] = (
        df['town'] + '_' + 
        df['flat_type'] + '_' + 
        df['remaining_lease_bucket'].astype(str)
    )
    
    for period in periods:
        col_name = f'avg_price_lag_{period}m'
        df[col_name] = df.groupby('group_key')['avg_price'].shift(period)
        logger.info(f"Added {col_name}")
    
    return df


def add_rolling_features(df, windows=[3, 6, 12]):

    
    for window in windows:
        col_name = f'avg_price_roll_{window}m'
        df[col_name] = (
            df.groupby('group_key')['avg_price']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        std_col_name = f'price_std_roll_{window}m'
        df[std_col_name] = (
            df.groupby('group_key')['avg_price']
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        
        logger.info(f"Added rolling features for {window} months")
    
    return df


def add_trend_features(df):
    logger.info("Adding trend features...")
    
    df['price_change_1m'] = df.groupby('group_key')['avg_price'].pct_change(1)
    
    df['price_change_3m'] = df.groupby('group_key')['avg_price'].pct_change(3)
    
    df['price_change_12m'] = df.groupby('group_key')['avg_price'].pct_change(12)
    
    df['price_change_from_start'] = (
        df.groupby('group_key')['avg_price']
        .transform(lambda x: (x - x.iloc[0]) / x.iloc[0] if len(x) > 0 else 0)
    )
    
    return df


def add_seasonality_features(df):
    """
    Add seasonality indicators
    """
    logger.info("Adding seasonality features...")
    
    df['month_num'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    df['year'] = df['month'].dt.year
    
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    
    return df


def add_categorical_encodings(df):
    """
    Encode categorical variables
    """
    logger.info("Adding categorical encodings...")
    
    town_encoding = df.groupby('town')['avg_price'].mean().to_dict()
    df['town_avg_price'] = df['town'].map(town_encoding)
    
    flat_type_order = {'2 ROOM': 1, '3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4, 'EXECUTIVE': 5}
    df['flat_type_encoded'] = df['flat_type'].map(flat_type_order).fillna(3)
    
    def lease_midpoint(bucket):
        if pd.isna(bucket):
            return 50
        if bucket == '0-20':
            return 10
        elif bucket == '20-40':
            return 30
        elif bucket == '40-60':
            return 50
        elif bucket == '60-80':
            return 70
        else:
            return 85
    
    df['lease_midpoint'] = df['remaining_lease_bucket'].apply(lease_midpoint)
    
    return df


def add_market_indicators(df):
    market_avg = df.groupby('month')['avg_price'].mean().reset_index()
    market_avg.columns = ['month', 'market_avg_price']
    df = df.merge(market_avg, on='month', how='left')
    
    df['price_vs_market'] = df['avg_price'] / df['market_avg_price']
    
    return df


def filter_for_modeling(df):
    """
    Filter and prepare data for modeling
    Keep only groups with sufficient history
    """
    
    group_counts = df.groupby('group_key').size().reset_index(name='obs_count')
    
    # Keep groups with at least 12 months of data
    valid_groups = group_counts[group_counts['obs_count'] >= 12]['group_key']
    
    df_filtered = df[df['group_key'].isin(valid_groups)].copy()
    
    logger.info(f"Kept {len(df_filtered)} rows from {len(valid_groups)} groups")
    logger.info(f"Removed {len(df) - len(df_filtered)} rows with insufficient history")
    
    return df_filtered


def upload_to_s3(df, bucket, key):
    logger.info(f"Uploading to s3://{bucket}/{key}")
    
    df['feature_engineering_timestamp'] = datetime.now()
    
    buffer = BytesIO()
    df.to_parquet(buffer, compression='snappy', index=False)
    
    buffer.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType='application/x-parquet'
    )
    
    logger.info(f"Successfully uploaded {df.shape[0]} rows, {df.shape[1]} features to S3")


def main():
    """
    Main execution function
    """
    try:
        df_bronze = read_from_s3(BUCKET_NAME, BRONZE_KEY)
        
        df_ts = create_time_series_groups(df_bronze)
        
        df_ts = add_lag_features(df_ts, periods=[1, 3, 6, 12])
        
        df_ts = add_rolling_features(df_ts, windows=[3, 6, 12])
        
        df_ts = add_trend_features(df_ts)
        
        df_ts = add_seasonality_features(df_ts)
        
        df_ts = add_categorical_encodings(df_ts)
        
        df_ts = add_market_indicators(df_ts)
        
        df_final = filter_for_modeling(df_ts)
        
        upload_to_s3(df_final, BUCKET_NAME, SILVER_KEY)
        
        logger.info("Silver layer feature engineering completed successfully!")
        
        logger.info(f"\nFeature Summary:")
        logger.info(f"Total features: {df_final.shape[1]}")
        logger.info(f"Unique groups: {df_final['group_key'].nunique()}")
        logger.info(f"Date range: {df_final['month'].min()} to {df_final['month'].max()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Silver layer feature engineering failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)