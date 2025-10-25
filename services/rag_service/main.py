from fastapi import FastAPI, UploadFile, File, HTTPException
from embeddings import embed_text_batch, embed_text
from vector_store import search_embeddings, store_embeddings
from utils import parse_content, chunk_text
from llm_service import generate_with_ollama

from pydantic import BaseModel
import time

app = FastAPI()

class SearchRequest(BaseModel):
    query : str

@app.get('/')
async def check_run():
    return {'status' : 'success', 'message': 'api running!'}

@app.post('/ingest/')
async def ingest_doc(file: UploadFile = File(...)):

    content = await file.read()
    text = parse_content(content)

    if not text or len(text) < 10:
        return {'status': 'error', 'message': 'Could not extract meaningful text from file'}
    
    chunks = chunk_text(text)

    embeddinngs = embed_text_batch(chunks)

    store_embeddings(embeddinngs, metadata={'filename' : file.filename, 'text_chunks': chunks})

    return {
        'status': 'success',
        'filename': file.filename,
        'chunks_created': len(chunks),
        'total_chars': len(text)
    }

@app.post('/search/')
async def search_doc(request: SearchRequest):
    query_embeddings = embed_text(request.query)
    results = search_embeddings(query_embeddings.tolist())

    return {'result' : results}

@app.post('/rag/')
async def rag_pipeline(request : SearchRequest):

    start = time.time()
    query = request.query

    try:

        query_embeddings = embed_text(query)
        res = search_embeddings(query_embeddings.tolist())

        context_parts = []

        for idx, result in enumerate(res):
            context_parts.append(
                f"[Source {idx} - {result['payload']['filename']}]:\n{result['payload']['text']}"
            )

        context = '\n\n'.join(context_parts)

        prompt = f"""You are a helpful AI assistant. Use the following context to answer the question accurately and precisely
        Context: {context}
        Question: {query}

        Instructions:
        - Answer based only on the provided context
        - If the context does not contain enough information, say so
        - Be cleat and Concise
        - Cite which source is used

        Answer:"""

        answer = generate_with_ollama(prompt)

        total_time = int((time.time() - start)*1000)

        result = {
            'query' : query,
            'answer' : answer,
            'respnse_time_ms' : total_time
        }

        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'RAG pipeline error: {str(e)}'
        )

@app.get('/health/')
async def health_check():
    health = {
        'api':'ok',
        'qdrant': 'unknown',
        'ollama' : 'unknown'
    }

    try:
        from vector_store import client
        client.get_collections()
        health['qdrant'] = 'ok'
    except:
        health['qdrant'] = 'error'

    try:
        import requests
        res = requests.get('http://localhost:11434/api/tags', timeout=3)
        if res.status_code == 200:
            health['ollama'] = 'ok'
    except:
        health['ollama'] = 'error'

    return health     
    