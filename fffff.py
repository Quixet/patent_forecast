import os
import time
import streamlit as st
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI

load_dotenv()
# Step 1: Authenticate and Initialize the Client
try:
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=os.environ["PROJECT_CONNECTION_STRING"],
    )
    print(" Client initialized successfully.")
except Exception as e:
    print(f" Error initializing client: {e}")
    exit()

# Step 2: Define Agent ID
agent_ids = ["asst_aqd20ehxXS7BblLSBZNtJIVU"]  # Replace with your actual agent ID
agents = []

# Step 3: Retrieve Agent
for agent_id in agent_ids:
    try:
        agent = project_client.agents.get_agent(agent_id)
        agents.append(agent)
        print(f" Agent retrieved: {agent.name}")
    except Exception as e:
        print(f" Error retrieving agent: {e}")

if not agents:
    print(" No agents retrieved. Exiting.")
    exit()

# Step 4: Create a Communication Thread
try:
    thread = project_client.agents.create_thread()
except Exception as e:
    print(f" Error creating thread: {e}")
    exit()

# Step 5: Send a Message to the Thread
user_message = "give me few publications related to CRAFT USING AN INERTIAL MASS REDUCTION DEVICE"

try:
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=user_message
    )
except Exception as e:
    print(f"Error sending message to thread: {e}")
    exit()

# Step 6: Process Agent Runs
for agent in agents:
    try:
        run = project_client.agents.create_and_process_run(
            thread_id=thread.id,
            agent_id=agent.id
        )
    except Exception as e:
        print(f"Error processing run for agent '{agent.name}': {e}")

# Step 7: Wait for the Agent to Process
time.sleep(10)  # Increase delay if needed

# Step 8: Retrieve and Display Messages in Correct Order
try:
    messages = project_client.agents.list_messages(thread_id=thread.id)

    if hasattr(messages, "data") and messages.data:
        # Sort messages by 'created_at' timestamp in ascending order
        sorted_messages = sorted(messages.data, key=lambda x: x.created_at)

        for msg in sorted_messages:
            if msg.content and isinstance(msg.content, list):
                for content_item in msg.content:
                    if content_item["type"] == "text":
                        print(f" {content_item['text']['value']}")
except Exception as e:
    print(f" Error retrieving messages: {e}")