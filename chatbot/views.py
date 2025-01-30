import redis
import pickle
from django.shortcuts import render
from django.http import JsonResponse
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from django.conf import settings

groq_api_key = settings.GROQ_API_KEY
google_api_key = settings.GOOGLE_API_KEY

# Initialize the model and prompt
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-it")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions:{input}
    """
)

# Redis connection handling
try:
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    r.ping()  # To check if the Redis server is alive
except redis.ConnectionError as e:
    print(f"Redis connection failed: {e}")
    r = None  # To indicate that Redis is unavailable

# Function to initialize the vector store
def initialize_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader("./test_pdf")
        docs = loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        final_documents = text_splitter.split_documents(docs)

        # Create FAISS vector store
        vectors = FAISS.from_documents(final_documents, embeddings)

        # Store vector store in Redis (if available) with 24-hour expiration
        if r:
            r.set('vector_store', pickle.dumps(vectors), ex=86400)
        return vectors  # return the vector store in case we need it elsewhere
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None  # Return None if there is an error

# Function to load the vector store from Redis or initialize it
def load_vector_store():
    if r:  # Check if Redis is available
        vectors = r.get('vector_store')
        if vectors:
            return pickle.loads(vectors)
        else:
            # If not found, initialize and store it
            print("Vector store not found in Redis, initializing it.")
            return initialize_vector_store()  # Initialize and return the vector store
    else:
        print("Redis is unavailable, initializing vector store without Redis.")
        return initialize_vector_store()  # Initialize the vector store without Redis

# Function to query the model with a prompt
def query_model(prompt1, vectors):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    try:
        # Set a timeout for the model query, e.g., 30 seconds
        response = retrieval_chain.invoke({'input': prompt1}, timeout=30)  # Timeout in seconds
    except Exception as e:
        print(f"Error during model query: {e}")
        response = {"answer": "Sorry, there was an issue processing your request."}  # Fallback error message
    return response

# View to query the chatbot
def chatbot_response(request):
    prompt1 = request.GET.get('input', '')  # Get input from frontend
    
    vectors = load_vector_store()  # Load vector store (Redis change)
    
    if prompt1:
        # Query the model with the loaded vectors
        response = query_model(prompt1, vectors)
        return JsonResponse({'answer': response['answer'], 'context': response.get('context', '')})
    
    return JsonResponse({"error": "No input provided"})

# # Home view to display the frontend interface (optional)
# def index(request):
#     return render(request, 'index.html')  # Ensure you have an 'index.html' in the templates folder
