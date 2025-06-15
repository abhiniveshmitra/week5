import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def get_embedding(text):
    res = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(res.data[0].embedding, dtype=np.float32)

def chunk_text(text, max_tokens=350):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i+max_tokens])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_embedding_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    return np.vstack(embeddings)

def cosine_sim(a, b):
    # a: [n, d], b: [d] or [m, d]
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def get_top_chunks(query, chunks, embedding_index, top_k=4):
    q_emb = get_embedding(query)
    sims = cosine_sim(embedding_index, q_emb)
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idxs]
