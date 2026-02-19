import streamlit as st
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests

st.title("RAG AI Teaching Assistant")
st.write("Ask questions about the MIT OCW course: Introduction to Computer Science and Programming in Python")

st.set_page_config(
    page_title="RAG AI Teaching Assistant",
    
    page_icon="🎓",
    layout="wide"
)






@st.cache_resource
def load_df():
    try:
        return joblib.load(r"C:\Users\Rajvardhan\OneDrive\Desktop\RAG-BASED-AI\embeddings.joblib")
    except Exception as e:
        st.error("Failed to load embeddings. Please regenerate embeddings.joblib.")
        st.exception(e)
        st.stop()



df = load_df()

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    r.raise_for_status()
    return r.json()["embeddings"]

def inference(prompt):
    r = requests.post("http://127.0.0.1:11434/api/generate", json={
        "model": "phi3",   # light model for 8GB RAM
        "prompt": prompt,
        "stream": False
    })
    r.raise_for_status()
    return r.json()["response"]

query = st.text_input(
    "Ask a question from the course:",
    placeholder="e.g. What are conditionals in Python?"
)



if st.button("Search"):
    
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching relevant video segments..."):
            question_embedding = create_embedding([query])[0]
            similarities = cosine_similarity(np.vstack(df["embedding"].values), [question_embedding]).flatten()
            top_k = 3
            max_indx = similarities.argsort()[::-1][:top_k]
            new_df = df.loc[max_indx]
            best_score = similarities[max_indx[0]]
            


            prompt = f"""
You are a teaching assistant for the free MIT OpenCourseWare course:
"Introduction to Computer Science and Programming in Python".

Use ONLY the provided video chunks to answer.

Answer in this format:

[Video]
File: <file>
Timestamp: <start>s - <end>s
Explanation: <direct answer>

Relevant Chunks:
{new_df[["file", "start", "end", "text"]].to_json(orient="records")}

User Question:
{query}
"""

            response = inference(prompt)

        st.subheader("Answer")
        st.write(response)
        st.caption(f"🔎 Top match confidence: {best_score:.2f}")


        st.subheader("Sources")
        for _, row in new_df.iterrows():
            st.markdown(f"""
**File:** {row['file']}  
**Timestamp:** {row['start']}s – {row['end']}s  
**Snippet:** {row['text']}
---
""")
