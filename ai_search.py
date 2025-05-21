from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search_client = SearchClient(
    endpoint="https://<your-service>.search.windows.net",
    index_name="<your-index-name>",
    credential=AzureKeyCredential("<your-key>")
)

query_text = "Яка різниця між дронами для розвідки та атаки?"

results = search_client.search(
    search_text=query_text,
    top=3,
    query_type="semantic",
    semantic_configuration_name="default"
)

top_chunks = [doc['content'] for doc in results]
context = "\n\n".join(top_chunks)
prompt = f"""

Context:
{context}

Question:
{query_text}
"""