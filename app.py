import streamlit as st
from youtube_ingest import get_transcript_chunks
from utils import create_embedding, inference
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib

# ---- fast embedding using parallel calls
def batch_embed(texts):
    def embed_one(text):
        return create_embedding(text)

    with ThreadPoolExecutor(max_workers=5) as executor:
        embeddings = list(executor.map(embed_one, texts))

    return embeddings


# ---- page setup
st.set_page_config(
    page_title="RAG AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("RAG AI Teaching Assistant")
st.write("Paste a YouTube link below and ask anything about the video.")

# ---- input
video_url = st.text_input("Enter YouTube Video URL")

# ---- process video
if st.button("Process Video"):
    if not video_url.strip():
        st.warning("Please enter a valid YouTube link.")
    else:
        video_key = hashlib.md5(video_url.encode()).hexdigest()

        if video_key in st.session_state:
            st.success("Video already processed. You can ask questions.")
        else:
            with st.spinner("Processing video..."):
                try:
                    chunks = get_transcript_chunks(video_url)
                    df = pd.DataFrame(chunks)

                    # speed optimizations
                    df = df.head(100)
                    df["text"] = df["text"].str[:120]

                    embeddings = batch_embed(df["text"].tolist())
                    df["embedding"] = embeddings

                    st.session_state["df"] = df
                    st.session_state[video_key] = df

                    st.success("Video is ready! Ask your question below.")

                except Exception as e:
                    st.error("Couldn't process this video. It may not have captions.")
                    st.exception(e)


# ---- question answering
df = st.session_state.get("df", None)

if df is not None:
    query = st.text_input(
        "Ask a question about the video:",
        placeholder="e.g. What are loops?"
    )

    if st.button("Search"):
        if not query.strip():
            st.warning("Type a question first 🙂")
        else:
            with st.spinner("Thinking..."):

                question_embedding = create_embedding(query)

                similarities = cosine_similarity(
                    np.vstack(df["embedding"].values),
                    [question_embedding]
                ).flatten()

                top_k = 3
                top_indices = similarities.argsort()[::-1][:top_k]
                results = df.loc[top_indices]
                confidence = similarities[top_indices[0]]

                prompt = f"""
You are a helpful teaching assistant.

Answer clearly and directly using ONLY the context provided.

Context:
{results[["file", "start", "end", "text"]].to_json(orient="records")}

Question:
{query}
"""

                answer = inference(prompt)

            st.subheader("Answer")
            st.write(answer)
            st.caption(f"Confidence score: {confidence:.2f}")

            st.subheader("Sources")
            for _, row in results.iterrows():
                st.markdown(f"""
**File:** {row['file']}  
**Time:** {row['start']}s – {row['end']}s  
**Snippet:** {row['text']}
---
""")

else:
    st.info("Start by pasting a YouTube link and clicking 'Process Video'.")