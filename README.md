# RAG Platform: Production-Grade Retrieval-Augmented Generation (RAG) System

A modern Retrieval-Augmented Generation backend designed for plug-and-play document QA, analytics, and evaluation in Python.

## Features

- **Document Ingestion:** PDF, TXT, and more; auto-chunked and embedded via sentence-transformers/all-MiniLM-L6-v2.

- **Vector DB:** Qdrant for storing semantic document chunks and supporting fast retrieval.

- **RAG Backend:** FastAPI-powered endpoints for ingestion, search, and end-to-end RAG Q&A.

- **LLM Support:** Local model serving via Ollama for answer generation.

- **Logging & Analytics:** Every query/answer/metric logged in PostgreSQL for traceability, error analysis, and future fine-tuning.

- **Retriever & Generator Evaluation:** RAG metrics to evaluate performance