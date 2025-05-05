import streamlit as st
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI

st.set_page_config(page_title="Azure AI Foundry Chat (No file reading)", layout="centered")

load_dotenv()

@st.cache_resource
def create_client():
    with AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=os.environ["PROJECT_CONNECTION_STRING"],
    ) as project_client:
        client: AzureOpenAI = project_client.inference.get_azure_openai_client(
            api_version=os.environ["AZURE_OPENAI_API_VERSION"]
        )
    return client


client = create_client()

def init_chat():
    st.session_state.agent = client.beta.assistants.create(
        model="gpt-4o-mini",
        name="streamlit-agent",
        instructions="""
You are a patent classification assistant. You will read a patent description and title and assign it to one of the following class class based on the context and keywords of the title and description:

1. Disposal
2. Compliance technology
3. Cannabis auxiliary medications
4. Animal
5. Sobriety test
6. Cannabinoid biosynthesis
7. Cultivation
8. Food/beverages/supplements
9. Devices
10. Processing
11. Medical
12. Compositions

Always return exactly one of the above class based on the context and keywords of the title and description.
After each classification, ask: \"Did I classify this correctly? (yes/no)\"
If the user replies \"no\", you must choose another class that fits the best after the previous one (do not repeat your self) and ask again.
If the user replies \"yes\", respond: \"Your data was stored in to blob storage.\"
Repeat asking user until user replies with \"yes\".
""",
    )
    st.session_state.thread = client.beta.threads.create()
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    init_chat()

st.title("Azure AI Foundry Chat (No file reading)")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter a massage"):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    client.beta.threads.messages.create(
        thread_id=st.session_state.thread.id,
        role="user",
        content=prompt,
    )

    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id=st.session_state.agent.id,
    )

    with st.spinner("Generate answer"):
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=run.id,
            )
            if run_status.status == "completed":
                break

    messages = client.beta.threads.messages.list(thread_id=st.session_state.thread.id)
    response = messages.data[0].content[0].text.value

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
