from typing import Tuple, List, Dict
import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from scripts.utils import load_dataset, load_vectorstore, get_top_related_questions, load_web_content, truncate_text
import logging
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize LangSmith
client = wrap_openai(openai.Client())

# Load your dataset
df = load_dataset('data/labeled_final.csv')

# Load the vector store
openai_api_key = os.getenv('OPENAI_API_KEY')
vectorstore = load_vectorstore('data', openai_api_key)

# Caching loaded web content
web_content_cache = {}

llm = ChatOpenAI(api_key=openai_api_key, model='gpt-3.5-turbo')

# Create a prompt template
template = """
You are an AI assistant specializing in property selling advice in the region of Australia. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Create a chain
chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def answer_question(user_question: str) -> Tuple[str, List[Dict[str, str]]]:
    # Get top related questions
    related_questions = get_top_related_questions(user_question, vectorstore)
    
    # Load content from related URLs and store contexts
    contexts = []
    full_context = ""
    for question in related_questions:
        url = df[df['question_text'] == question]['question_url'].values[0]
        context = load_web_content(url, web_content_cache)
        contexts.append({"question": question, "url": url})
        full_context += " ".join([doc.page_content for doc in context]) + "\n\n"
    
    # Log similar questions and their contexts to a file
    log_data = {
        "user_question": user_question,
        "similar_questions": [{"question": c["question"], "url": c["url"]} for c in contexts]
    }
    with open('log.json', 'a') as f:
        json.dump(log_data, f, indent=4)
        f.write('\n')
    
    # Truncate the full context
    truncated_context = truncate_text(full_context)
    
    # Generate answer
    response = chain.invoke({"context": truncated_context, "question": user_question})
    
    return response, [{"question": c["question"], "url": c["url"]} for c in contexts]