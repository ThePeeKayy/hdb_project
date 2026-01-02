"""
Silver Layer: Feature Engineering and Aggregation
Aggregates bronze data by month, region, and flat_type
Saves to S3 for gold layer consumption
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BUCKET_NAME = "hdb-prediction-pipeline"
BRONZE_KEY = "bronze/resale-data.parquet"
SILVER_KEY = "silver/features.parquet"

s3_client = boto3.client("s3")


def read_from_s3(bucket, key):
    """Read parquet from S3"""
    logger.info(f"Reading s3://{bucket}/{key}")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(BytesIO(response["Body"].read()))
    logger.info(f"✓ Loaded {len(df):,} rows")
    return df


def extract_storey_mid(storey_range):
    """Extract middle value from storey range (e.g., '10 TO 12' -> 11.0)"""
    try:
        parts = str(storey_range).split(" TO ")
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) / 2
        return np.nan
    except:
        return np.nan


def process_bronze_data(df):
    """Process and clean bronze data"""
    logger.info("Processing bronze data...")
    
    
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    
    
    df["storey_mid"] = df["storey_range"].apply(extract_storey_mid)
    
    
    df["remaining_lease"] = (99 - (df["month"].dt.year - df["lease_commence_date"])).clip(lower=0)
    
    
    numeric_cols = ["resale_price", "floor_area_sqm", "storey_mid", "remaining_lease", "lease_commence_date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    
    initial_count = len(df)
    df = df.dropna(subset=["month", "region", "flat_type", "resale_price", "floor_area_sqm"])
    dropped = initial_count - len(df)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped:,} rows with null values")
    
    logger.info(f"✓ After processing: {len(df):,} rows")
    
    return df


def aggregate_data(df):
    """Aggregate by month, region, and flat_type"""
    logger.info("Aggregating data...")
    
    agg_df = df.groupby(["month", "region", "flat_type"]).agg(
        avg_price=("resale_price", "mean"),
        avg_floor_area=("floor_area_sqm", "mean"),
        avg_storey=("storey_mid", "mean"),
        avg_remaining_lease=("remaining_lease", "mean"),
        avg_lease_commence=("lease_commence_date", "mean"),
        count=("resale_price", "count")
    ).reset_index()
    
    logger.info(f"✓ Aggregated to {len(agg_df):,} rows")
    logger.info(f"✓ Groups: {agg_df.groupby(['region', 'flat_type']).ngroups}")
    
    
    for col in ["avg_storey", "avg_remaining_lease", "avg_lease_commence"]:
        
        agg_df[col] = agg_df.groupby(["region", "flat_type"])[col].transform(
            lambda x: x.fillna(x.mean())
        )
        
        if agg_df[col].isna().any():
            global_mean = agg_df[col].mean()
            agg_df[col] = agg_df[col].fillna(global_mean)
    
    
    agg_df["group_key"] = agg_df["region"] + "_" + agg_df["flat_type"]
    
    logger.info(f"✓ Unique groups: {agg_df['group_key'].nunique()}")
    logger.info(f"✓ Group keys: {sorted(agg_df['group_key'].unique())}")
    
    return agg_df


def save_to_s3(df, bucket, key):
    """Save dataframe to S3 as parquet"""
    logger.info(f"Saving to s3://{bucket}/{key}")
    
    buffer = BytesIO()
    df.to_parquet(buffer, compression='snappy', index=False)
    
    buffer.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType='application/x-parquet'
    )
    
    logger.info(f"✓ Saved {len(df):,} rows to S3")


def main():
    """Main execution"""
    try:
        logger.info("=" * 60)
        logger.info("SILVER LAYER: Feature Engineering")
        logger.info("=" * 60)
        
        
        df = read_from_s3(BUCKET_NAME, BRONZE_KEY)
        
        if len(df) == 0:
            logger.error("Bronze layer is empty!")
            return False
        
        logger.info(f"Date range: {df['month'].min()} to {df['month'].max()}")
        logger.info(f"Regions: {df['region'].unique()}")
        logger.info(f"Flat types: {df['flat_type'].unique()}")
        
        
        df = process_bronze_data(df)
        
        if len(df) == 0:
            logger.error("All data filtered out during processing!")
            return False
        
        
        agg_df = aggregate_data(df)
        
        if len(agg_df) == 0:
            logger.error("Aggregation resulted in empty dataframe!")
            return False
        
        
        save_to_s3(agg_df, BUCKET_NAME, SILVER_KEY)
        
        
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total aggregated rows: {len(agg_df):,}")
        logger.info(f"Date range: {agg_df['month'].min()} to {agg_df['month'].max()}")
        logger.info(f"Unique groups: {agg_df['group_key'].nunique()}")
        logger.info(f"\nRegion distribution:")
        logger.info(agg_df['region'].value_counts().to_string())
        logger.info(f"\nFlat type distribution:")
        logger.info(agg_df['flat_type'].value_counts().to_string())
        logger.info("=" * 60)
        logger.info("✓ Silver layer completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Silver layer failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)