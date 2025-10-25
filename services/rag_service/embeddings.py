from sentence_transformers import SentenceTransformer
from typing import List

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    embeddings = model.encode([text], convert_to_numpy=True).flatten()
    return embeddings

def embed_text_batch(texts: List[str]):
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return [embed.tolist() for embed in embeddings]