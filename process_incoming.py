import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    r.raise_for_status()
    return r.json()["embeddings"]

def inference(prompt):
    r = requests.post("http://127.0.0.1:11434/api/generate", json={
        "model": "phi3",   # lighter model for 8GB RAM
        "prompt": prompt,
        "stream": False
    })
    r.raise_for_status()
    return r.json()["response"]

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df["embedding"].values), [question_embedding]).flatten()
top_results = 5
max_indx = similarities.argsort()[::-1][:top_results]
new_df = df.loc[max_indx]

prompt = f"""
You are a teaching assistant for the free MIT OpenCourseWare course:
"Introduction to Computer Science and Programming in Python".

You are given relevant subtitle chunks from the course videos.
Each chunk contains:
- file (video filename)
- start time (seconds)
- end time (seconds)
- transcript text

STRICT OUTPUT FORMAT:
For each relevant result, answer in the following format exactly:

[Video]
File: <video filename>
Timestamp: <start_time>s - <end_time>s
Explanation: <directly answer the user's question using the transcript, not describing the transcript>

RULES:
- Do NOT invent timestamps or file names.
- Use ONLY the provided chunks.
- Do NOT describe the transcript (e.g., "the transcript suggests..." or "this segment discusses...").
- Answer the user's question directly, then use the video and timestamp as the source.
- If multiple chunks are relevant, list them in separate blocks.
- If the question is unrelated to this course, say:
  "I can only answer questions related to the MIT OpenCourseWare Python course."

-------------------------
Relevant Video Chunks:
{new_df[["file", "start", "end", "text"]].to_json(orient="records")}
-------------------------

User Question:
{incoming_query}

Now provide the answer in the strict format above:
"""

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

response = inference(prompt)
print(response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)
