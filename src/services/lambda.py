import json
import boto3
import requests
from datetime import datetime
import pickle
import numpy as np
import uuid
import time

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('listings')

def load_model():
    with open("arima_unified_model.pkl", 'rb') as f:
        return pickle.load(f)

model_data = load_model()

def fetch_hdb_data_all(town=None, flat_type=None, max_records=10000):
    dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    base_url = "https://data.gov.sg/api/action/datastore_search"

    filters = {}
    if town:
        filters["town"] = town
    if flat_type:
        filters["flat_type"] = flat_type

    all_records = []
    limit = 1000
    offset = 0

    while offset < max_records:
        params = {
            "resource_id": dataset_id,
            "filters": json.dumps(filters),
            "limit": limit,
            "offset": offset
        }
        r = requests.get(base_url, params=params)
        r.raise_for_status()
        result = r.json()["result"]
        records = result.get("records", [])
        if not records:
            break

        all_records.extend(records)
        if len(records) < limit:
            break
        offset += limit

    return all_records

def filter_by_remaining_lease_bucket(records, bucket_str, current_year=None):
    if current_year is None:
        current_year = datetime.now().year

    min_years, max_years = map(int, bucket_str.split('-'))
    filtered = []
    for r in records:
        lease_year = int(r["lease_commence_date"])
        remaining_lease = 99 - (current_year - lease_year)
        if min_years <= remaining_lease < max_years:
            r["remaining_lease_estimate"] = remaining_lease
            filtered.append(r)

    return filtered

def lambda_handler(event, context):
    method = event['httpMethod']
    path = event['path']
    
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Content-Type': 'application/json'
    }
    
    if method == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}
    
    if method == 'GET' and path == '/api/listings':
        response = table.scan()
        items = response['Items']
        
        formatted_items = []
        for item in items:
            formatted_item = {
                'id': item.get('id'),
                'title': item.get('title'),
                'description': item.get('description'),
                'imageUrl': item.get('imageUrl'),
                'date': item.get('date'),
                'datetime': item.get('datetime'),
                'category': item.get('category'),
                'location': item.get('location'),
                'author': {
                    'name': item.get('authorName'),
                    'telegram': item.get('authorTelegram')
                }
            }
            formatted_items.append(formatted_item)
        
        return {'statusCode': 200, 'headers': headers, 'body': json.dumps(formatted_items)}
    
    elif method == 'POST' and path == '/api/listings':
        body = json.loads(event['body'])
        
        listing_id = str(uuid.uuid4())
        ttl = int(time.time()) + (30 * 24 * 60 * 60)
        
        item = {
            'id': listing_id,
            'title': body.get('title', ''),
            'description': body.get('description', ''),
            'imageUrl': body.get('imageUrl', ''),
            'date': body.get('date', datetime.now().strftime('%b %d, %Y')),
            'datetime': body.get('datetime', datetime.now().strftime('%Y-%m-%d')),
            'category': body.get('category', ''),
            'location': body.get('location', ''),
            'authorName': body.get('author', {}).get('name', '') if isinstance(body.get('author'), dict) else '',
            'authorTelegram': body.get('author', {}).get('telegram', '') if isinstance(body.get('author'), dict) else '',
            'ttl': ttl
        }
        
        table.put_item(Item=item)
        
        return {'statusCode': 201, 'headers': headers, 'body': json.dumps({'message': 'Listing created successfully', 'id': listing_id})}
    
    elif method == 'POST' and path == '/api/predict':
        if model_data is None:
            return {'statusCode': 500, 'headers': headers, 'body': json.dumps({"error": "Prediction model not loaded"})}
            
        body = json.loads(event['body'])
        town = body.get('town')
        flat_type = body.get('flat_type')
        remaining_lease_bucket = body.get('remaining_lease_bucket')

        all_records = fetch_hdb_data_all(town=town, flat_type=flat_type, max_records=10000)
        filtered_records = filter_by_remaining_lease_bucket(all_records, remaining_lease_bucket)
        
        if len(filtered_records) == 0:
            result = {
                "current_avg_price": 0,
                "predicted_6m_price": 0,
                "predicted_12m_price": 0,
                "trend": 'No sale'
            }
        else:
            current_avg_price = sum(int(r['resale_price']) for r in filtered_records) / len(filtered_records)
            town_encoded = model_data['le_town'].transform([town])[0]
            flat_type_encoded = model_data['le_flat'].transform([flat_type])[0]
            lease_bucket_encoded = model_data['le_lease'].transform([remaining_lease_bucket])[0]
            
            exog_pred = np.column_stack([
                [town_encoded] * 12,
                [flat_type_encoded] * 12,
                [lease_bucket_encoded] * 12
            ])
            
            forecast = model_data['model_fit'].forecast(steps=12, exog=exog_pred)
            predicted_6m_price = float(forecast[5])
            predicted_12m_price = float(forecast[11])
            
            change_percent = ((predicted_12m_price - current_avg_price) / current_avg_price) * 100
            if change_percent > 2:
                trend = 'increasing'
            elif change_percent < -2:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            result = {
                "current_avg_price": current_avg_price,
                "predicted_6m_price": predicted_6m_price,
                "predicted_12m_price": predicted_12m_price,
                "trend": trend
            }
        
        return {'statusCode': 200, 'headers': headers, 'body': json.dumps(result)}
    
    elif method == 'POST' and path == '/api/helper':
        body = json.loads(event['body'])
        prompt = body.get('prompt', '')
        
        response = f"HDB advice for: {prompt}. This endpoint will be enhanced with full PDF processing."
        
        return {'statusCode': 200, 'headers': headers, 'body': json.dumps({'advice': response})}
    
    return {'statusCode': 404, 'headers': headers, 'body': json.dumps({'error': 'Route not found'})}