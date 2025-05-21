import os
import uuid
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")

TXT_FOLDER = "txt_files"

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)


def get_embedding(text: str):
    response = client.embeddings.create(
        input=[text],
        model=OPENAI_DEPLOYMENT
    )
    return response.data[0].embedding


documents_to_upload = []

for filename in os.listdir(TXT_FOLDER):
    if filename.endswith(".txt"):
        file_path = os.path.join(TXT_FOLDER, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            continue

        chunk_id = str(uuid.uuid4())
        title = os.path.splitext(filename)[0]
        vector = get_embedding(content)

        doc = {
            "chunk_id": chunk_id,
            "parent_id": "1000",
            "chunk": content,
            "title": title,
            "group": "group1",
            "text_vector": vector
        }

        documents_to_upload.append(doc)

if documents_to_upload:
    result = search_client.upload_documents(documents=documents_to_upload)
    print(" Success:", len(result))
else:
    print("No files found .txt")