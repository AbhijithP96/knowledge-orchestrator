from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from embeddings import embed_text_batch, embed_text
from vector_store import search_embeddings, store_embeddings
from utils import parse_content, chunk_text
from llm_service import generate_with_ollama
from database import get_db, Document, QueryLog, EvaluationResult
from eval import RetrievalMetrics, GeneratorMetrics

from pydantic import BaseModel
import time

app = FastAPI()

class SearchRequest(BaseModel):
    query : str

class FeebackRequest(BaseModel):
    query_log_id: int
    feedback: str
    comments: str = None

@app.get('/')
async def check_run():
    return {'status' : 'success', 'message': 'api running!'}

@app.post('/ingest/')
async def ingest_doc(file: UploadFile = File(...), db: Session = Depends(get_db)):

    content = await file.read()
    text = parse_content(content)

    if not text or len(text) < 10:
        return {'status': 'error', 'message': 'Could not extract meaningful text from file'}
    
    chunks = chunk_text(text)

    embeddinngs = embed_text_batch(chunks)

    store_embeddings(embeddinngs, metadata={'filename' : file.filename, 'text_chunks': chunks})

    document_log = Document(
        filename = file.filename,
        content_type = file.content_type,
        file_size = len(content),
        processed = True,
        num_chunks = len(chunks)
    )

    db.add(document_log)
    db.commit()
    db.refresh(document_log)

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
async def rag_pipeline(request : SearchRequest, db: Session = Depends(get_db)):

    start = time.time()
    query = request.query
    rm = RetrievalMetrics()
    gm = GeneratorMetrics()

    try:

        query_embeddings = embed_text(query)
        res = search_embeddings(query_embeddings.tolist(), limit=15)

        retrieved_texts =[r['payload']['text'] for r in res]

        eval_metrics = {
            'recall_at_k' : rm.calculate_recall_at_k(query, retrieved_texts),
            'redundancy_rate' : rm.calculate_redundancy_rate(retrieved_texts),
            'avg_precision' : rm.calculate_avg_precsion(query, retrieved_texts)
        } 

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

        Answer:"""

        answer = generate_with_ollama(prompt)

        total_time = int((time.time() - start)*1000)

        ragas_scores = gm.evaluate_with_ragas(
            query=query,
            answer=answer,
            context=retrieved_texts
        )

        query_log = QueryLog(
            query_text = query,
            response = answer,
            response_time_ms = total_time,

            evaluation_metric = eval_metrics
        )

        db.add(query_log)
        db.commit()
        db.refresh(query_log)


        evaluation_log = EvaluationResult(
            query_log_id = query_log.id,
            faithfulness = ragas_scores['faithfulness'],
            answer_relevancy = ragas_scores['answer_relevancy'],
            context_precision = ragas_scores.get('context_precision', 0.0),
            context_recall = ragas_scores.get('context_recall',0.0)
        )

        db.add(evaluation_log)
        db.commit()
        db.refresh(evaluation_log)

        result = {
            'query' : query,
            'answer' : answer,
            'respnse_time_ms' : total_time,
            'query_log_id' : query_log.id
        }

        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'RAG pipeline error: {str(e)}'
        )

@app.post('/feedback/')
async def submit_feedback(request: FeebackRequest, db: Session = Depends(get_db)):
    query_log = db.query(QueryLog).filter(QueryLog.id == request.query_log_id).first()

    if not query_log:
        raise HTTPException(status_code=404, detail='Query not found.')
    
    query_log.user_feedback = request.feedback
    db.commit()

    return {
        'status':'success',
        'query_log_id':request.query_log_id,
        'feedback':request.feedback
    }

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
    