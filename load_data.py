import pymysql  # або import psycopg2 для PostgreSQL
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

connection = pymysql.connect(
    host="your-db-host.rds.amazonaws.com",
    user="your-username",
    password="your-password",
    database="your-database",
    port=5432
)

with connection.cursor() as cursor:
    cursor.execute("SELECT id, title, content FROM patents LIMIT 1000;")
    rows = cursor.fetchall()

documents = []
for row in rows:
    doc = {
        "id": str(row[0]),
        "title": row[1],
        "content": row[2]
    }
    documents.append(doc)

search_service_name = "your-search-service-name"
index_name = "your-index-name"
api_key = "your-search-api-key"

endpoint = f"https://{search_service_name}.search.windows.net"
client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

result = client.upload_documents(documents)
print(f"Upload status: {result}")
