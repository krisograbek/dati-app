import os
import streamlit as st
from dotenv import load_dotenv
from sql_helpers import ChatbotPipeline

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
    # Note: Initialization of the pipeline is moved inside the main block

# Only proceed with chat functionality if a CSV file has been uploaded and processed
if csv_file is not None and openai_api_key:
    # Initialize the ChatbotPipeline with the uploaded CSV file
    pipeline = ChatbotPipeline(csv_file=csv_file)

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new user questions
    if user_prompt := st.chat_input("Your question about the CSV:"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = pipeline.run_full_chain(user_prompt)  # Use the pipeline's method
            message_placeholder.markdown(str(response))

        st.session_state.messages.append(
            {"role": "assistant", "content": str(response)}
        )

else:
    st.write("Please upload a CSV file to proceed.")
