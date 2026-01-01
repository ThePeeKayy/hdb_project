"""
Bronze Layer: Data Ingestion from data.gov.sg
Runs on EC2 t4g.micro
Execution time: ~2 minutes
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


def fetch_all_records(limit=100000):
    all_records = []
    offset = 0
    
    while True:
        try:
            params = {
                'resource_id': RESOURCE_ID,
                'limit': 100,  # API limit per request
                'offset': offset
            }
            
            response = requests.get(DATA_GOV_API, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('success'):
                logger.error(f"API returned unsuccessful response: {data}")
                break
                
            records = data.get('result', {}).get('records', [])
            
            if not records:
                break
                
            all_records.extend(records)
            logger.info(f"Fetched {len(all_records)} records so far...")
            
            # Check if we've reached the limit
            if len(records) < 100 or len(all_records) >= limit:
                break
                
            offset += 100
            
        except Exception as e:
            logger.error(f"Error fetching data at offset {offset}: {e}")
            break
    
    logger.info(f"Total records fetched: {len(all_records)}")
    return all_records


def clean_and_validate_data(records):
    df = pd.DataFrame(records)
    
    logger.info(f"Initial shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    required_cols = [
        'month', 'town', 'flat_type', 'block', 'street_name',
        'storey_range', 'floor_area_sqm', 'flat_model', 
        'lease_commence_date', 'remaining_lease', 'resale_price'
    ]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    if 'resale_price' in df.columns:
        df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
    
    if 'floor_area_sqm' in df.columns:
        df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    
    if 'lease_commence_date' in df.columns:
        df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
    
    if 'month' in df.columns:
        df['month'] = pd.to_datetime(df['month'], errors='coerce')
    
    critical_fields = ['month', 'town', 'flat_type', 'resale_price']
    df = df.dropna(subset=[col for col in critical_fields if col in df.columns])
    
    logger.info(f"After cleaning shape: {df.shape}")
    
    return df


def calculate_remaining_lease_years(df):
    if 'remaining_lease' not in df.columns:
        logger.warning("remaining_lease column not found")
        return df
    
    def parse_lease(lease_str):
        if pd.isna(lease_str):
            return None
        
        try:
            lease_str = str(lease_str).lower()
            years = 0
            months = 0
            
            if 'year' in lease_str:
                years = int(lease_str.split('year')[0].strip())
            
            if 'month' in lease_str:
                month_part = lease_str.split('year')[-1] if 'year' in lease_str else lease_str
                months = int(month_part.split('month')[0].strip())
            
            return years + (months / 12)
        except:
            return None
    
    df['remaining_lease_years'] = df['remaining_lease'].apply(parse_lease)
    
    return df


def add_lease_buckets(df):
    if 'remaining_lease_years' not in df.columns:
        return df
    
    def assign_bucket(years):
        if pd.isna(years):
            return None
        if years < 20:
            return '0-20'
        elif years < 40:
            return '20-40'
        elif years < 60:
            return '40-60'
        elif years < 80:
            return '60-80'
        else:
            return '80+'
    
    df['remaining_lease_bucket'] = df['remaining_lease_years'].apply(assign_bucket)
    
    return df


def upload_to_s3(df, bucket, key):
    table = pa.Table.from_pandas(df)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression='snappy')
    
    buffer.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType='application/x-parquet'
    )
    
    logger.info(f"Successfully uploaded {df.shape[0]} rows to S3")


def main():
    """
    Main execution function
    """
    try:
        records = fetch_all_records()
        
        if not records:
            logger.error("No records fetched. Exiting.")
            return False
        
        df = clean_and_validate_data(records)
        
        df = calculate_remaining_lease_years(df)
        
        df = add_lease_buckets(df)
        
        df['ingestion_timestamp'] = datetime.now()
        df['ingestion_date'] = datetime.now().strftime('%Y-%m-%d')
        
        upload_to_s3(df, BUCKET_NAME, BRONZE_KEY)
        
        logger.info("Bronze layer ingestion completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Bronze layer ingestion failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)