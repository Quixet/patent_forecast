import os
import fitz
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import pytesseract

POPPLER_BIN_PATH = r"C:\Users\Yurii\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text.strip()


def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang='eng') + "\n"
    return text.strip()

pdf_folder = "pdfs"
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

texts = []
file_names = []

for file in pdf_files:
    full_path = os.path.join(pdf_folder, file)
    text = extract_text_from_pdf(full_path)
    if text and text.strip():
        texts.append(text)
        file_names.append(file)


model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts)

scaler = StandardScaler()
vectors_scaled = scaler.fit_transform(vectors)

for file in pdf_files:
    full_path = os.path.join(pdf_folder, file)
    text = extract_text_from_pdf(full_path)
    print(f"{file}: {len(text.strip())} символів")


if len(texts) < 3:
    print("Недостатньо валідних PDF-файлів з текстом для кластеризації.")
    exit()

clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
labels = clusterer.fit_predict(vectors_scaled)

umap_model = umap.UMAP(random_state=42)
embedding = umap_model.fit_transform(vectors_scaled)

plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
for i, label in enumerate(unique_labels):
    idx = np.where(labels == label)
    plt.scatter(embedding[idx, 0], embedding[idx, 1], label=f"Cluster {label}", color=colors[i])
plt.title("Patent Clustering with HDBSCAN")
plt.legend()
plt.grid(True)
plt.show()

df = pd.DataFrame({
    "File": file_names,
    "Cluster": labels,
    "Text": texts
})
print(df[["File", "Cluster"]])

df.to_csv("clustered_patents.csv", index=False, encoding="utf-8")