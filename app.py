import streamlit as st
from scripts.question_answering import answer_question
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
