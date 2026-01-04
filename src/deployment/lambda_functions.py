import json
import boto3
from datetime import datetime
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