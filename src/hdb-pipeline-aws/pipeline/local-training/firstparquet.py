"""
Initial Setup: Combine local CSVs into Bronze Parquet
Combines all CSV files in current directory into single parquet file
Uploads to S3 with proper structure matching the incremental script
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from datetime import datetime
import logging
from io import BytesIO
from glob import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BUCKET_NAME = 'hdb-prediction-pipeline'
BRONZE_KEY = 'bronze/resale-data.parquet'

s3_client = boto3.client('s3')


def map_town_to_region(town):
    """Map town to Singapore region"""
    central = ['BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA', 'GEYLANG',
               'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'TOA PAYOH']
    east = ['BEDOK', 'PASIR RIS', 'TAMPINES']
    north = ['ANG MO KIO', 'HOUGANG', 'PUNGGOL', 'SENGKANG', 'SERANGOON',
             'SEMBAWANG', 'WOODLANDS', 'YISHUN']
    west = ['BUKIT BATOK', 'BUKIT PANJANG', 'CHOA CHU KANG', 'CLEMENTI',
            'JURONG EAST', 'JURONG WEST', 'LIM CHU KANG']

    if town in central:
        return 'CENTRAL'
    elif town in east:
        return 'EAST'
    elif town in north:
        return 'NORTH'
    elif town in west:
        return 'WEST'
    else:
        return 'OTHERS'


def combine_csvs(pattern='*.csv'):
    """Combine all CSV files matching pattern"""
    csv_files = glob(pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {pattern}")
    
    logger.info(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        logger.info(f"  - {f}")
    
    # Read and combine all CSVs
    dfs = []
    for csv_file in csv_files:
        logger.info(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file)
        logger.info(f"  Loaded {len(df):,} rows")
        dfs.append(df)
    
    # Concatenate all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined total: {len(combined):,} rows")
    
    return combined


def clean_and_validate_data(df):
    """Clean data - same as bronze script"""
    logger.info(f"Initial shape: {df.shape}")
    
    # Convert data types
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
    df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    
    # Drop nulls
    df = df.dropna(subset=['month', 'town', 'flat_type', 'resale_price', 'floor_area_sqm'])
    
    # Filter outliers
    df = df[(df['resale_price'] > 50000) & (df['resale_price'] < 2000000)]
    df = df[(df['floor_area_sqm'] > 20) & (df['floor_area_sqm'] < 300)]
    
    logger.info(f"After cleaning: {len(df):,} rows")
    
    # Add region column
    df['region'] = df['town'].apply(map_town_to_region)
    
    logger.info(f"Region distribution:\n{df['region'].value_counts()}")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(
        subset=['month', 'town', 'flat_type', 'block', 'street_name', 'storey_range', 'floor_area_sqm'],
        keep='last'
    )
    if len(df) < initial_count:
        logger.info(f"Removed {initial_count - len(df):,} duplicate rows")
    
    return df


def save_local_parquet(df, filename='resale-data.parquet'):
    """Save as parquet file locally"""
    # Convert all object columns to string to avoid type issues
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    df.to_parquet(filename, compression='snappy', index=False, engine='pyarrow')
    logger.info(f"✓ Saved parquet file to {filename}")


def main():
    """Main execution"""
    try:
        logger.info("=" * 60)
        logger.info("INITIAL SETUP: Combine CSVs -> Parquet")
        logger.info("=" * 60)
        
        # Step 1: Combine all CSVs
        combined_df = combine_csvs('*.csv')
        
        # Step 2: Clean and validate
        clean_df = clean_and_validate_data(combined_df)
        
        # Step 3: Add metadata
        clean_df['ingestion_timestamp'] = datetime.now()
        
        # Step 4: Save as parquet locally
        save_local_parquet(clean_df)
        
        # Summary
        logger.info("=" * 60)
        logger.info("✓ Setup completed successfully!")
        logger.info(f"  Total records: {len(clean_df):,}")
        logger.info(f"  Date range: {clean_df['month'].min()} to {clean_df['month'].max()}")
        logger.info(f"  Saved to: resale-data.parquet")
        logger.info("  Manually upload this to S3 at s3://hdb-prediction-pipeline/bronze/")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Setup failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)