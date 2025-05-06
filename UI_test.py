import streamlit as st
import requests
import time

# Налаштування
AZURE_ENDPOINT = "https://ai-patentforecast-playground999373195511.services.ai.azure.com/models"
API_VERSION = "2024-12-01-preview"
ANALYZER_ID = "patent_categorisation"
SUBSCRIPTION_KEY = "встав_тут_свій_ключ"  # заміни на справжній ключ

def get_analyze_url():
    return f"{AZURE_ENDPOINT}/contentunderstanding/analyzers/{ANALYZER_ID}:analyze?api-version={API_VERSION}"

def begin_analysis_with_file(file_bytes):
    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY,
        "x-ms-useragent": "cu-sample-ui"
    }
    response = requests.post(get_analyze_url(), headers=headers, data=file_bytes)
    response.raise_for_status()
    return response.headers["operation-location"]

def poll_result(operation_location: str):
    headers = {
        "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY
    }
    start = time.time()
    while True:
        res = requests.get(operation_location, headers=headers)
        res.raise_for_status()
        result = res.json()
        status = result.get("status", "").lower()
        if status == "succeeded":
            return result
        elif status == "failed":
            raise RuntimeError("Аналіз не вдався.")
        time.sleep(1)
        if time.time() - start > 60:
            raise TimeoutError("Аналіз перевищив ліміт часу")

def extract_class_label(result: dict) -> str:
    """
    Витягує назву класу із результату аналізу.
    Структура result залежить від конфігурації analyzer.
    Тут приклад для predictions[0]["class"]
    """
    try:
        return result["result"]["predictions"][0]["class"]
    except Exception:
        return "Не вдалося визначити клас із результату."

# Streamlit UI
st.title("Patent Classification (Local File Upload)")

uploaded_file = st.file_uploader("Завантаж .txt або .pdf файл для аналізу", type=["txt", "pdf"])

if uploaded_file is not None:
    if st.button("Аналізувати"):
        try:
            file_bytes = uploaded_file.read()  # НЕ декодуємо
            with st.spinner("Аналіз виконується..."):
                operation_location = begin_analysis_with_file(file_bytes)
                result = poll_result(operation_location)
                class_label = extract_class_label(result)
            st.success(f"🧠 Клас визначено: {class_label}")
        except Exception as e:
            st.error(f"❌ Помилка: {e}")
