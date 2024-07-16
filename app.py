import streamlit as st
from scripts.question_answering import answer_question
# from dotenv import load_dotenv
import os

langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
langchain_project = os.getenv('LANGCHAIN_PROJECT')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Example of using these variables in LangChain setup
# langchain.setup(endpoint=langchain_endpoint, api_key=langchain_api_key, project=langchain_project, tracing_v2=langchain_tracing_v2)

# # Load environment variables
# load_dotenv()

st.title("Property Selling Advice Bot")

user_question = st.text_input("Ask a question about property selling:")

if st.button("Get Advice"):
    if user_question:
        answer, similar_questions = answer_question(user_question)
        
        st.write("## User Question:")
        st.write(user_question)
        
        st.write("## Final Answer:")
        st.write(answer)
        
        st.write("## Similar Questions and URLs:")
        for i, sq in enumerate(similar_questions, 1):
            st.write(f"### {i}. Similar Question: {sq['question']}")
            st.write(f"URL: {sq['url']}")
    else:
        st.write("Please enter a question.")
