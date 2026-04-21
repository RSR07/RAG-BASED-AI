import requests

# create embedding
def create_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]

# generate response
def inference(prompt):
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]