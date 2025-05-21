import streamlit as st
import os
import requests
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

# Load environment variables
load_dotenv()

# Config
st.set_page_config(page_title="Azure AI Foundry Chat", layout="centered")

# Endpoint configuration
ENDPOINT_URL = os.getenv("PROMPT_FLOW_ENDPOINT_URL")
USE_API_KEY = True
API_KEY = os.getenv("PROMPT_FLOW_API_KEY")

# Setup headers for authentication
def get_auth_headers():
    headers = {
        "Content-Type": "application/json"
    }
    if USE_API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    else:
        token = DefaultAzureCredential().get_token("https://ml.azure.com/.default")
        headers["Authorization"] = f"Bearer {token.token}"
    return headers

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "group_input" not in st.session_state:
    st.session_state.group_input = ""

st.title("Azure AI Foundry Chat (Endpoint-Based)")

# Input for group
st.session_state.group_input = st.text_input(
    "Enter group name (e.g., group3):",
    value=st.session_state.group_input,
    placeholder="group3"
)

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Prompt input
if prompt := st.chat_input("Enter a message"):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    payload = {
        "group_input": st.session_state.group_input,
        "chat_input": prompt,
        "chat_history": []
    }

    headers = get_auth_headers()

    with st.spinner("Generating response..."):
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()

        # Extract result — adjust if necessary based on actual response
        assistant_message = result.get("chat_output", "No response received.")

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
        st.chat_message("assistant").markdown(assistant_message)
    else:
        error_msg = f"Error: {response.status_code} - {response.text}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        st.chat_message("assistant").markdown(error_msg)
