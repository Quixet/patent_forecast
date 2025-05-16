# file for testing prompt flow



import os
import requests
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

# Replace with your actual endpoint and deployment name
ENDPOINT_URL = os.getenv("PROMPT_FLOW_ENDPOINT_URL")  # e.g. "https://<your-endpoint>.eastus.inference.ml.azure.com/score"
USE_API_KEY = True  # Set to False if you're using RBAC authentication
API_KEY = os.getenv("PROMPT_FLOW_API_KEY")

# Prepare your input payload (depends on your prompt flow schema)
payload = {
    "group_input": "group3",
    "chat_input": "who is the inventor of patent number US 10 , 144 , 532 B2?",
    "chat_history":[]
}

# Authenticate using API Key or Azure AD
headers = {
    "Content-Type": "application/json"
}

if USE_API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"  # From Azure ML online endpoint
else:
    # Use Azure AD token with DefaultAzureCredential
    credential = DefaultAzureCredential()
    token = credential.get_token("https://ml.azure.com/.default")
    headers["Authorization"] = f"Bearer {token.token}"

# Make the request
response = requests.post(ENDPOINT_URL, headers=headers, json=payload)

if response.status_code == 200:
    result = response.json()
    print("Response from Prompt Flow:")
    print(result)
else:
    print(f"Failed to call Prompt Flow: {response.status_code}")
    print(response.text)
