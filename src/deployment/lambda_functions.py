import json
import boto3
import requests
from datetime import datetime
import pickle
import numpy as np
import uuid
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

os.environ["AWS_DEFAULT_REGION"] = "ap-southeast-1" 

dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-1')
table = dynamodb.Table('listings')

model_data = None
vector_db = None
embeddings = None

def load_model():
    global model_data
    if model_data is None:
        try:
            model_path = 'arima_unified_model.pkl'
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            model_data = None
    return model_data

def load_vector_db():
    global vector_db, embeddings
    if vector_db is None:
        try:
            embeddings = OpenAIEmbeddings()
            pdf_path = 'HDBregulations.pdf'
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100
            )
            docs = text_splitter.split_documents(pages)
            vector_db = FAISS.from_documents(docs, embeddings)
        except Exception as e:
            print(f"Error loading vector database: {e}")
            vector_db = None
    return vector_db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    llm = ChatOpenAI(model="gpt-4o")
    
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""You are an expert HDB resale assistant with deep knowledge of Singapore's HDB regulations and policies.

        Your task is to provide accurate, helpful answers based strictly on the official HDB regulations provided below.

        INSTRUCTIONS:
        - Provide a concise answer in 5-10 sentences maximum
        - Focus on the most important information only
        - Cite specific regulations when possible
        - If information is not found in the provided documents, state this clearly
        - Be direct and practical

        QUESTION: {question}

        RELEVANT HDB REGULATIONS:
        {docs}

        ANSWER:"""
    )
    
    chain = prompt | llm
    response = chain.invoke({"question": query, "docs": docs_page_content})
    
    response_content = response.content if hasattr(response, 'content') else str(response)
    response_content = response_content.replace("\n", "")
    return response_content

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
        try:
            response = table.scan()
            items = response['Items']
            
            formatted_items = []
            for item in items:
                formatted_item = {
                    'id': item.get('id'),
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'imageUrl': item.get('imageUrl', ''),
                    'date': item.get('date', ''),
                    'datetime': item.get('datetime', ''),
                    'category': item.get('category', ''),
                    'location': item.get('location', ''),
                    'author': {
                        'name': item.get('authorName', ''),
                        'telegram': item.get('authorTelegram', '')
                    }
                }
                formatted_items.append(formatted_item)
            
            return {'statusCode': 200, 'headers': headers, 'body': json.dumps(formatted_items)}
        except Exception as e:
            return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(e)})}
    
    elif method == 'POST' and path == '/api/listings':
        try:
            body = json.loads(event['body'])
            
            listing_id = str(uuid.uuid4())
            ttl = int(time.time()) + (30 * 24 * 60 * 60) 
            
            author = body.get('author', {})
            if isinstance(author, dict):
                author_name = author.get('name', '')
                author_telegram = author.get('telegram', '')
            else:
                author_name = ''
                author_telegram = ''
            
            item = {
                'id': listing_id,
                'title': body.get('title', ''),
                'description': body.get('description', ''),
                'imageUrl': body.get('imageUrl', ''), 
                'date': body.get('date', datetime.now().strftime('%b %d, %Y')),
                'datetime': body.get('datetime', datetime.now().strftime('%Y-%m-%d')),
                'category': body.get('category', ''),
                'location': body.get('location', ''),
                'authorName': author_name,
                'authorTelegram': author_telegram,
                'ttl': ttl
            }

            print(item)

            
            table.put_item(Item=item)
            
            return {'statusCode': 201, 'headers': headers, 'body': json.dumps({'message': 'Listing created successfully', 'id': listing_id})}
        except Exception as e:
            print(e)
            return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(e)})}
    
    elif method == 'POST' and path == '/api/predict':
        try:
            current_model = load_model()
            if current_model is None:
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
                town_encoded = current_model['le_town'].transform([town])[0]
                flat_type_encoded = current_model['le_flat'].transform([flat_type])[0]
                lease_bucket_encoded = current_model['le_lease'].transform([remaining_lease_bucket])[0]
                
                exog_pred = np.column_stack([
                    [town_encoded] * 12,
                    [flat_type_encoded] * 12,
                    [lease_bucket_encoded] * 12
                ])
                
                forecast = current_model['model_fit'].forecast(steps=12, exog=exog_pred)
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
        except Exception as e:
            return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(e)})}
    
    elif method == 'POST' and path == '/api/helper':
        try:
            db = load_vector_db()
            if db is None:
                return {'statusCode': 500, 'headers': headers, 'body': json.dumps({"error": "Vector database not loaded"})}
                
            body = json.loads(event['body'])
            prompt = body.get('prompt', '')
            
            response = get_response_from_query(db, prompt)
            
            return {'statusCode': 200, 'headers': headers, 'body': json.dumps({'advice': response})}
        except Exception as e:
            return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(e)})}
    
    return {'statusCode': 404, 'headers': headers, 'body': json.dumps({'error': 'Route not found'})}