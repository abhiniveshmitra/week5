# src/document_processor.py
import os
import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader

@st.cache_resource
def get_embeddings_model():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("ADA_DEPLOYMENT_NAME"),
        openai_api_version="2024-05-01-preview",
    )

def get_text_from_files(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif uploaded_file.name.endswith('.txt'):
            text += uploaded_file.read().decode('utf-8')
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

@st.cache_resource(show_spinner="Creating document embeddings...")
def create_vector_store(_chunks):
    embeddings = get_embeddings_model()
    return FAISS.from_texts(texts=_chunks, embedding=embeddings)
