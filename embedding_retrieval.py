import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

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
    # Simple whitespace chunking; for prod use tiktoken for token-accurate
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

def get_top_chunks(query, chunks, embedding_index, top_k=4):
    q_emb = get_embedding(query)
    sims = cosine_similarity([q_emb], embedding_index)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idxs]
