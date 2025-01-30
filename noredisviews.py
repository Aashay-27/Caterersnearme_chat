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

# Load API keys from environment variables
from django.conf import settings

groq_api_key = settings.GROQ_API_KEY
google_api_key = settings.GOOGLE_API_KEY

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

# Function to initialize the vector store
def initialize_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./test_pdf")
    docs = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    final_documents = text_splitter.split_documents(docs)

    # Create FAISS vector store
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

# Function to query the model
def query_model(prompt1, vectors):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({'input': prompt1})
    return response

# View to query the chatbot
def chatbot_response(request):
    prompt1 = request.GET.get('input', '')  # Get input from frontend
    
    # Check if vector store is initialized in the session
    if 'vectors' not in request.session:
        # Initialize the vector store and store it in the session
        vectors = initialize_vector_store()
        request.session['vectors'] = vectors
    
    # Retrieve the vector store from session
    vectors = request.session.get('vectors')

    if prompt1:
        # Query the model
        response = query_model(prompt1, vectors)
        return JsonResponse({'answer': response['answer'], 'context': response['context']})
    
    return JsonResponse({"error": "No input provided"})
