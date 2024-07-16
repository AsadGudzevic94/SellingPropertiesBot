import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from typing import List, Dict
import pickle
import logging
import os
# from dotenv import load_dotenv
import faiss

# Load environment variables
# load_dotenv()

langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
langchain_project = os.getenv('LANGCHAIN_PROJECT')
openai_api_key = os.getenv('OPENAI_API_KEY')

def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['question_text'] = df['question_text'].astype(str)
    return df[df['question_text'] != 'nan']

def load_vectorstore(load_path: str, openai_api_key: str) -> FAISS:
    # Load the FAISS index
    index = faiss.read_index(os.path.join(load_path, "faiss_index.bin"))
    
    # Load the metadata
    with open(os.path.join(load_path, "faiss_meta.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    # Create the docstore and the id mapping
    docstore = InMemoryDocstore({i: Document(page_content=text) for i, text in enumerate(metadata["texts"])})
    index_to_docstore_id = {i: i for i in range(len(metadata["texts"]))}
    
    # Create FAISS vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    
    return vectorstore

def get_top_related_questions(query: str, vectorstore: FAISS, top_k: int = 3) -> List[str]:
    related = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in related]

def load_web_content(url: str, cache: Dict[str, List[str]]) -> List[str]:
    if url in cache:
        return cache[url]
    
    loader = WebBaseLoader(url)
    try:
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_data = text_splitter.split_documents(data)
        cache[url] = split_data
        return split_data
    except Exception as e:
        logging.error(f"Error loading content from {url}: {e}")
        return []

def truncate_text(text: str, max_tokens: int = 128000) -> str:
    words = text.split()
    truncated_words = words[:max_tokens * 3 // 4]  # Approximate words to tokens ratio
    return " ".join(truncated_words)
