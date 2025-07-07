import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()
embeddings = OpenAIEmbeddings()

def create_vector_db_from_pdf(pdf_path: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(pages)
    db = FAISS.from_documents(docs, embeddings)
    return db

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


@app.route('/api/helper', methods=['POST'])
def get_advice():
    data = request.get_json()
    prompt = data.get('prompt')
    db = create_vector_db_from_pdf('HDBregulations.pdf')
    response = get_response_from_query(db, prompt)
    return jsonify({'advice': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)