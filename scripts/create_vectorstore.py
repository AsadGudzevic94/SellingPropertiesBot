import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pickle  # Importing pickle
# from dotenv import load_dotenv
import faiss  # Importing faiss

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

def create_and_save_vectorstore(df: pd.DataFrame, save_path: str, openai_api_key: str):
    texts = df['question_text'].tolist()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    # Save the FAISS index and the metadata separately
    faiss.write_index(vectorstore.index, os.path.join(save_path, "faiss_index.bin"))
    with open(os.path.join(save_path, "faiss_meta.pkl"), "wb") as f:
        pickle.dump({"texts": texts}, f)

if __name__ == "__main__":
    df = load_dataset('data/labeled_final.csv')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    create_and_save_vectorstore(df,"data", openai_api_key)
