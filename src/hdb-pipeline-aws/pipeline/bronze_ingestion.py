"""
Bronze Layer: Data Ingestion from data.gov.sg
Matches ULTRA-SIMPLE NHiTS model structure
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


def map_town_to_region(town):
    """Map town to Singapore region - EXACT match with NHiTS training"""
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


def fetch_all_records():
    """Fetch all records from data.gov.sg API"""
    all_records = []
    offset = 0
    
    while True:
        try:
            params = {
                'resource_id': RESOURCE_ID,
                'limit': 100,
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
                
            offset += 100
            
        except Exception as e:
            logger.error(f"Error fetching data at offset {offset}: {e}")
            break
    
    logger.info(f"Total records fetched: {len(all_records)}")
    return all_records


def clean_and_validate_data(records):
    """Clean data - EXACT match with NHiTS training cleaning"""
    df = pd.DataFrame(records)
    
    logger.info(f"Initial shape: {df.shape}")
    
    
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
    df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    
    
    df = df.dropna(subset=['month', 'town', 'flat_type', 'resale_price', 'floor_area_sqm'])
    df = df[(df['resale_price'] > 50000) & (df['resale_price'] < 2000000)]
    df = df[(df['floor_area_sqm'] > 20) & (df['floor_area_sqm'] < 300)]
    
    logger.info(f"After cleaning: {len(df):,} rows")
    
    
    df['region'] = df['town'].apply(map_town_to_region)
    
    logger.info(f"Region distribution:\n{df['region'].value_counts()}")
    
    return df


def upload_to_s3(df, bucket, key):
    """Upload to S3 as Parquet"""
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
    
    logger.info(f"✓ Uploaded {df.shape[0]:,} rows to s3://{bucket}/{key}")


def main():
    """Main execution"""
    try:
        logger.info("=" * 60)
        logger.info("BRONZE LAYER: Data Ingestion")
        logger.info("=" * 60)
        
        
        records = fetch_all_records()
        
        if not records:
            logger.error("No records fetched. Exiting.")
            return False
        
        
        df = clean_and_validate_data(records)
        
        
        df['ingestion_timestamp'] = datetime.now()
        
        
        upload_to_s3(df, BUCKET_NAME, BRONZE_KEY)
        
        logger.info("✓ Bronze layer completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Bronze layer failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)