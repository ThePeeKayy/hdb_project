"""
Bronze Layer: Simple Current Month Check
Just checks if current month exists in parquet, if not, appends it
Runs monthly via cron on EC2 t4g.micro
"""

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from datetime import datetime
import logging
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BUCKET_NAME = 'hdb-prediction-pipeline'
DATA_GOV_API = 'https://data.gov.sg/api/action/datastore_search'
RESOURCE_ID = 'd_8b84c4ee58e3cfc0ece0d773c8ca6abc'
BRONZE_KEY = 'bronze/resale-data.parquet'

s3_client = boto3.client('s3')


def map_town_to_region(town):
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


def get_current_month_string():
    now = datetime.now()
    return now.strftime('%Y-%m')


def load_parquet_from_s3(bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    buffer = BytesIO(response['Body'].read())
    df = pd.read_parquet(buffer)
    logger.info(f"Loaded {len(df):,} rows from S3")
    
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    
    current_month = datetime.now()
    five_years_ago = current_month.replace(year=current_month.year - 5)
    
    original_count = len(df)
    df = df[df['month'] >= five_years_ago]
    removed_count = original_count - len(df)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count:,} rows older than {five_years_ago.strftime('%Y-%m')}")
    
    return df


def month_exists_in_df(df, target_month):
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    existing_months = df['month'].dt.strftime('%Y-%m').unique()
    exists = target_month in existing_months
    logger.info(f"Month {target_month} exists: {exists}")
    return exists


def fetch_current_month_from_api(target_month):
    logger.info(f"Fetching records for {target_month}")
    
    params = {
        'resource_id': RESOURCE_ID,
        'limit': 10000,
        'sort': 'month desc'
    }
    
    resp = requests.get(DATA_GOV_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    if not data.get('success'):
        raise RuntimeError(f"API failed: {data}")
    
    records = data['result']['records']
    filtered = [r for r in records if r.get('month', '').startswith(target_month)]
    
    logger.info(f"Found {len(filtered)} records for {target_month}")
    return filtered


def clean_data(records):
    df = pd.DataFrame(records)
    
    if df.empty:
        return df
    
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
    df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    
    if 'lease_commence_date' in df.columns:
        df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce').astype('Int64')
    
    df = df.dropna(subset=['month', 'town', 'flat_type', 'resale_price', 'floor_area_sqm'])
    df = df[(df['resale_price'] > 50000) & (df['resale_price'] < 2000000)]
    df = df[(df['floor_area_sqm'] > 20) & (df['floor_area_sqm'] < 300)]
    
    df['region'] = df['town'].apply(map_town_to_region)
    
    return df

def upload_to_s3(df, bucket, key):
    df_copy = df.copy()
    
    if 'lease_commence_date' in df_copy.columns:
        df_copy['lease_commence_date'] = pd.to_numeric(df_copy['lease_commence_date'], errors='coerce').astype('Int64')
    
    if 'remaining_lease' in df_copy.columns:
        df_copy['remaining_lease'] = df_copy['remaining_lease'].astype(str)
    
    table = pa.Table.from_pandas(df_copy)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression='snappy')
    
    buffer.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType='application/x-parquet'
    )
    
    logger.info(f"Uploaded {len(df):,} rows to s3://{bucket}/{key}")


def main():
    try:
        current_month = get_current_month_string()
        logger.info(f"Current month: {current_month}")
        
        existing_df = load_parquet_from_s3(BUCKET_NAME, BRONZE_KEY)
        
        if month_exists_in_df(existing_df, current_month):
            logger.info(f"Month {current_month} already exists. Nothing to do.")
            return existing_df
        
        records = fetch_current_month_from_api(current_month)
        
        if not records:
            logger.warning(f"No data available for {current_month}")
            return existing_df
        
        new_df = clean_data(records)
        
        if new_df.empty:
            logger.warning("No valid records after cleaning")
            return existing_df
        
        if 'ingestion_timestamp' in existing_df.columns:
            existing_df = existing_df.drop(columns=['ingestion_timestamp'])
        
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df['ingestion_timestamp'] = datetime.now()
        
        upload_to_s3(combined_df, BUCKET_NAME, BRONZE_KEY)
        
        logger.info(f"Added {len(new_df):,} rows for {current_month}. Total: {len(combined_df):,}")
        return combined_df
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    df = main()
    exit(0)