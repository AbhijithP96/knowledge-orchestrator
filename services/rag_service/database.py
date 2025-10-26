from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

DATABASE_URL = f'postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@localhost:5432/{os.environ['POSTGRES_DB']}'

engine = create_engine(DATABASE_URL)
session = sessionmaker(autoflush=False, autocommit=False, bind=engine)
Base = declarative_base()

def default_query_eval_data():

    return {
        'recall_at_k' : 0,
        'redundacy_rate' : 0,
        'avg_precision': 0,
    }

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String)
    file_size = Column(Integer)
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    num_chunks = Column(Integer)
    qdrant_collection = Column(String, default="documents")

class QueryLog(Base):
    __tablename__ = 'queries'

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Integer)
    user_feedback = Column(String)

    evaluation_metric = Column(JSONB, default=default_query_eval_data)

class EvaluationResult(Base):

    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    query_log_id = Column(Integer)
    faithfulness = Column(Float)
    answer_relevancy = Column(Float)
    context_precision = Column(Float)
    context_recall = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = session()
    try:
        yield db
    finally:
        db.close()

