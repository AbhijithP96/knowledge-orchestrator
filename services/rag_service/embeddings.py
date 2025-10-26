from sentence_transformers import SentenceTransformer
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer(os.environ['EMBED_MODEL'])

def embed_text(text: str):
    embeddings = model.encode([text], convert_to_numpy=True).flatten()
    return embeddings

def embed_text_batch(texts: List[str]):
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return [embed.tolist() for embed in embeddings]