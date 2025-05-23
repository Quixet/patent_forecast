import os
import time
import streamlit as st
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI

load_dotenv()



@st.cache_resource
def create_client():
    project_client = AIProjectClient.from_connection_string(
        conn_str=os.environ["PROJECT_CONNECTION_STRING"],
        credential=DefaultAzureCredential(),
    )
    return project_client.inference.get_azure_openai_client(
        api_version=os.environ["AZURE_OPENAI_API_VERSION"]
    )

client = create_client()



def init_chat():
    st.session_state.agent = client.beta.assistants.create(
        model="gpt-4o-mini",
        name="streamlit-agent-4o-mini",
        instructions="""
You are a patent classification assistant. You will read a patent description and title and assign it to one of the following class based on the context and keywords of the title and description:

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
If the user replies \"no\", choose another class that fits the best after the previous one (do not repeat your self) and ask again.
If the user replies \"yes\", respond: \"Your data was stored in to blob storage.\"
Repeat asking user until user replies with \"yes\".
"""
    )
    st.session_state.thread = client.beta.threads.create()
    st.session_state.chat_history = []
    st.session_state.thread_files = []
    st.session_state.uploaded_text = ""
    st.session_state.last_prompt = ""

if "agent" not in st.session_state:
    init_chat()

st.sidebar.title("Settings")

if st.sidebar.button("Clean the chat"):
    init_chat()
    st.experimental_rerun()

uploaded_files = st.sidebar.file_uploader(
    "\U0001F4CE Upload .txt file (will be used silently)",
    type=["txt"],
    accept_multiple_files=False,
    label_visibility="visible"
)

if uploaded_files:
    text_from_file = uploaded_files.read().decode("utf-8")
    st.session_state.uploaded_text = text_from_file
    st.sidebar.success(f"File '{uploaded_files.name}' was loaded.")

st.title("Azure AI Foundry Chat")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Enter a message")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Construct final prompt (append file text if available)
    if st.session_state.uploaded_text:
        full_prompt = f"""User message:
{user_input}

Patent description:
{st.session_state.uploaded_text.strip()}
"""
    else:
        full_prompt = user_input

    client.beta.threads.messages.create(
        thread_id=st.session_state.thread.id,
        role="user",
        content=full_prompt
    )

    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id=st.session_state.agent.id
    )

    with st.spinner("Generate answer"):
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=st.session_state.thread.id)
    assistant_response = next(
        (m.content[0].text.value for m in reversed(messages.data) if m.role == "assistant"),
        "The agent did not answer."
    )

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)