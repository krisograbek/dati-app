import os
import streamlit as st

import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up the Streamlit page
st.set_page_config(page_title="Ask your CSV")
st.title("Ask your CSV Chatbot 📈")

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for CSV upload and database initialization
with st.sidebar:
    st.write("## Upload CSV Data")
    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    # openai_api_key = st.text_input('OpenAI API Key', type='password')
    engine = None
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        engine = create_engine("sqlite:///dati.db")
        df.to_sql("dati", engine, index=False, if_exists="replace")

# Only proceed with chat functionality if a CSV file has been uploaded and processed
# if engine is not None and openai_api_key:
if engine is not None:
    db = SQLDatabase(engine=engine)
    llm = ChatOpenAI(
        model="gpt-4-0125-preview", temperature=0, openai_api_key=openai_api_key
    )
    agent_executor = create_sql_agent(
        llm, db=db, agent_type="openai-tools", verbose=True
    )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new user questions
    # user_prompt = st.chat_input("Your question about the CSV:")
    if user_prompt := st.chat_input("Your question about the CSV:"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # Assuming agent_executor can handle session_state.messages format or adapt as needed
            response = agent_executor.invoke(
                user_prompt
            )  # Adjust this call based on your agent's method
            # Ensure the response is processed as needed to be displayed correctly
            message_placeholder.markdown(str(response["output"]))

        st.session_state.messages.append(
            {"role": "assistant", "content": str(response["output"])}
        )


else:
    st.write("Please upload a CSV file to proceed.")
