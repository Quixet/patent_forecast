import os
import streamlit as st
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI

load_dotenv()

with AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
) as project_client:

    client:AzureOpenAI = project_client.inference.get_azure_openai_client(
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )
    with client:
        agents_response = project_client.agents.list_agents()
        agents = agents_response.get("data", [])
        for agent in agents:
            print(f"Deleting agent: {agent['id']} ({agent['name']})")
            project_client.agents.delete_agent(agent["id"])
            # client.beta.assistants.delete(agent.id)

        # client.beta.threads.delete("your-thread-id")

        files_response = client.files.list()
        for file in files_response.data:
            print(f"Deleting file: {file.id} ({file.filename})")
            client.files.delete(file.id)

        print("All agents and files deleted.")
