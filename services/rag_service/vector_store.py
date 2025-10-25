from qdrant_client import QdrantClient
import uuid

client = QdrantClient(host='localhost', port=6333)

COLLECTION_NAME = 'documents'

def store_embeddings(embeddings, metadata):

    points = []
    chunks = metadata['text_chunks']

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        id = str(uuid.uuid4())
        
        payload = {
            'filename' : metadata['filename'],
            'chunk_index': idx,
            'text': chunk,
            'chunk_length' : len(chunk)
        }

        points.append({
            'id' : id,
            'vector':embedding,
            'payload' : payload
        })

    client.upsert(COLLECTION_NAME, 
                  points= points,
                  wait=True)
    
def search_embeddings(query_embeddings, limit=3):
    result = client.search(COLLECTION_NAME,
                           query_vector= query_embeddings,
                           with_payload=True,
                           with_vectors=False,
                           limit=limit)
    
    return [{
        'id' : res.id,
        'score' : res.score,
        'payload':res.payload
    } for res in result]