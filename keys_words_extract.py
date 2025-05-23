from keybert import KeyBERT


with open("txt_files/descriptin_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

kw_model = KeyBERT('paraphrase-MiniLM-L6-v2')
keywords = kw_model.extract_keywords(text, top_n=10)
print(keywords)