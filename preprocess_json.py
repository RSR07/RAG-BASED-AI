import requests
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    r.raise_for_status()
    return r.json()["embeddings"]

json_dir = r"C:\Users\Rajvardhan\OneDrive\Desktop\RAG-BASED-AI\json"
json_files = os.listdir(json_dir)

my_dicts = []
chunk_id = 0

for json_file in json_files:
    with open(os.path.join(json_dir, json_file), encoding="utf-8") as f:
        content = json.load(f)

    print(f"Creating embeddings for {json_file}")
    texts = [c["text"] for c in content["chunks"]]
    embeddings = create_embedding(texts)

    for i, chunk in enumerate(content["chunks"]):
        my_dicts.append({
            "file": json_file,
            "chunk_id": chunk_id,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "embedding": embeddings[i]
        })
        chunk_id += 1

 # remove later to process all files

df = pd.DataFrame(my_dicts)

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

emb_matrix = np.vstack(df["embedding"].values)
similarities = cosine_similarity(emb_matrix, [question_embedding]).flatten()

top_k = 3
top_idx = similarities.argsort()[::-1][:top_k]

results = df.iloc[top_idx]
print(results[["file", "chunk_id", "start", "end", "text"]])
joblib.dump(df, "embeddings.joblib")
print("Saved embeddings.joblib")


df["file"] = df["file"].astype(str)
df["text"] = df["text"].astype(str)


joblib.dump(df, "embeddings.joblib")


