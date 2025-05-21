import os
import fitz
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_BIN_PATH = r"C:\Users\Yurii\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

def extract_text_smart(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    if text.strip():
        return text.strip(), "text"

    print(f"OCR для: {os.path.basename(pdf_path)}")
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_BIN_PATH)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img, lang="eng") + "\n"
        return ocr_text.strip(), "ocr"
    except Exception as e:
        print(f"Помилка OCR у {pdf_path}: {e}")
        return "", "error"

pdf_folder = "pdfs"
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

texts = []
file_names = []
sources = []

for file in pdf_files:
    full_path = os.path.join(pdf_folder, file)
    text, source = extract_text_smart(full_path)
    if text:
        texts.append(text)
        file_names.append(file)
        sources.append(source)

if len(texts) < 3:
    print("Недостатньо валідних PDF-файлів для кластеризації.")
    exit()

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts)

scaler = StandardScaler()
vectors_scaled = scaler.fit_transform(vectors)

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
plt.title("Patent Clustering (Text + OCR)")
plt.legend()
plt.grid(True)
plt.show()

df = pd.DataFrame({
    "File": file_names,
    "Cluster": labels,
    "Source": sources,
    "Text": texts
})
df.to_csv("clustered_patents_combined.csv", index=False, encoding="utf-8")
print(df[["File", "Cluster", "Source"]])
