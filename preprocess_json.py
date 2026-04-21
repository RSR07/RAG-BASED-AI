import requests
import os
import json
import numpy as np
import pandas as pd
import joblib
import faiss


# ---- Simple helper to get embeddings from Ollama
def create_embedding(text_list):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    response.raise_for_status()
    return response.json()["embeddings"]


# ---- Folder where your transcript JSON files are stored
json_dir = r"C:\Users\Rajvardhan\OneDrive\Desktop\RAG-BASED-AI\json"
json_files = os.listdir(json_dir)

all_rows = []
chunk_id = 0


# ---- Go through each file and generate embeddings
for json_file in json_files:
    file_path = os.path.join(json_dir, json_file)

    with open(file_path, encoding="utf-8") as f:
        content = json.load(f)

    print(f"\nProcessing: {json_file}")

    texts = [chunk["text"] for chunk in content["chunks"]]

    # Generate embeddings in one go (you can batch this later if needed)
    embeddings = create_embedding(texts)

    for i, chunk in enumerate(content["chunks"]):
        all_rows.append({
            "file": json_file,
            "chunk_id": chunk_id,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "embedding": embeddings[i]
        })
        chunk_id += 1


# ---- Convert everything into a DataFrame
df = pd.DataFrame(all_rows)

print("\nCreating FAISS index...")

# Convert embeddings to numpy matrix
emb_matrix = np.vstack(df["embedding"].values).astype("float32")

# Build FAISS index (L2 distance)
index = faiss.IndexFlatL2(emb_matrix.shape[1])
index.add(emb_matrix)

# ---- Save both data and index
faiss.write_index(index, "faiss.index")
joblib.dump(df, "embeddings.joblib")

print("Done! Embeddings and FAISS index are ready.")