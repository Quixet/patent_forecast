import os
import time
import streamlit as st
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI
import fitz  # PyMuPDF

load_dotenv()

st.sidebar.title("Settings")

# Initialize uploaded_text only once
if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""

uploaded_files = st.sidebar.file_uploader(
    "\U0001F4CE Upload .pdf file",
    type=["pdf"],
    accept_multiple_files=False,
    label_visibility="visible"
)

# When a file is uploaded, extract and store its text
if uploaded_files:
    st.sidebar.success(f"File '{uploaded_files.name}' was loaded.")
    
    # Read PDF content using fitz
    with fitz.open(stream=uploaded_files.read(), filetype="pdf") as doc:
        extracted_text = ""
        for page in doc:
            extracted_text += page.get_text()
    
        # Store extracted text in session state
        st.session_state.uploaded_text = extracted_text

        st.write(extracted_text) 

st.title("Admin (create session)")

# Optional: Display the extracted text
if st.session_state.uploaded_text:
    st.subheader("Extracted Text:")
    st.text_area("Text from PDF:", st.session_state.uploaded_text, height=300)
