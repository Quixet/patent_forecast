from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import os


def cu_call(question: str) -> dict:
    client = AIProjectClient.from_connection_string(
        conn_str=os.environ["PROJECT_CONNECTION_STRING"],
        credential=DefaultAzureCredential()
    )

    cu_client = client.content_understanding

    result = cu_client.analyze_text(
        text=question,
        project_name="назва_твого_CU_проекту",
        deployment_name="default"
    )

    return {
        "categories": result.categories,
        "entities": result.entities,
        "summaries": result.summaries
    }

text = "descriptin_text.txt"
cu_call(text)