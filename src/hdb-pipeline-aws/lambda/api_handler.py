"""
Lambda Function: api-handler
Handles API requests for HDB predictions
Reads from DynamoDB cache (fast) or S3 (fallback)
"""

import json
import boto3
import logging
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)


dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')


DYNAMODB_TABLE = 'hdb-predictions'
BUCKET_NAME = 'hdb-prediction-pipeline'
PREDICTIONS_KEY = 'gold/predictions-cache.parquet'
METRICS_KEY = 'metrics/model-performance.json'

table = dynamodb.Table(DYNAMODB_TABLE)


class DecimalEncoder(json.JSONEncoder):
    """Helper to convert Decimal to float for JSON"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


def get_prediction_from_dynamodb(town, flat_type, lease_bucket):
    """
    Retrieve prediction from DynamoDB cache
    """
    try:
        group_key = f"{town}_{flat_type}_{lease_bucket}"
        
        response = table.get_item(
            Key={'town_flattype_lease': group_key}
        )
        
        if 'Item' in response:
            item = response['Item']
            return {
                'current_avg_price': float(item['current_avg_price']),
                'predicted_6m_price': float(item['predicted_6m_price']),
                'predicted_12m_price': float(item['predicted_12m_price']),
                'trend': item['trend'],
                'confidence_score': float(item.get('confidence_score', 0.85)),
                'last_updated': item.get('last_updated', ''),
                'source': 'cache'
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error reading from DynamoDB: {e}")
        return None


def get_prediction_from_s3(town, flat_type, lease_bucket):
    try:
        import pandas as pd
        from io import BytesIO
        
        logger.info("Falling back to S3 for prediction")
        
        
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=PREDICTIONS_KEY)
        buffer = BytesIO(response['Body'].read())
        df = pd.read_parquet(buffer)
        
        
        result = df[
            (df['town'] == town) &
            (df['flat_type'] == flat_type) &
            (df['remaining_lease_bucket'] == lease_bucket)
        ]
        
        if len(result) > 0:
            row = result.iloc[0]
            return {
                'current_avg_price': float(row['current_avg_price']),
                'predicted_6m_price': float(row['predicted_6m_price']),
                'predicted_12m_price': float(row['predicted_12m_price']),
                'trend': row['trend'],
                'confidence_score': float(row.get('confidence_score', 0.85)),
                'last_updated': row.get('last_updated', ''),
                'source': 's3'
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error reading from S3: {e}")
        return None


def get_model_metrics():
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=METRICS_KEY)
        metrics = json.loads(response['Body'].read())
        return metrics
        
    except Exception as e:
        logger.error(f"Error reading metrics: {e}")
        return None


def lambda_handler(event, context):
    try:
        
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/predict')
        
        if path == '/health':
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'status': 'healthy'})
            }
        
        if path == '/metrics':
            metrics = get_model_metrics()
            
            if metrics:
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps(metrics, cls=DecimalEncoder)
                }
            else:
                return {
                    'statusCode': 404,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({'error': 'Metrics not found'})
                }
        
        if path == '/predict':
            
            if http_method == 'POST':
                body = json.loads(event.get('body', '{}'))
            else:  
                body = event.get('queryStringParameters', {})
            
            town = body.get('town', '').upper()
            flat_type = body.get('flat_type', '').upper()
            lease_bucket = body.get('remaining_lease_bucket', '')
            
            
            if not all([town, flat_type, lease_bucket]):
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'error': 'Missing required parameters: town, flat_type, remaining_lease_bucket'
                    })
                }
            
            
            prediction = get_prediction_from_dynamodb(town, flat_type, lease_bucket)
            
            
            if not prediction:
                prediction = get_prediction_from_s3(town, flat_type, lease_bucket)
            
            
            if prediction:
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps(prediction, cls=DecimalEncoder)
                }
            else:
                
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'current_avg_price': 0,
                        'predicted_6m_price': 0,
                        'predicted_12m_price': 0,
                        'trend': 'No sale',
                        'message': 'Insufficient historical data for this combination'
                    })
                }
        
        
        return {
            'statusCode': 404,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': 'Not found'})
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }