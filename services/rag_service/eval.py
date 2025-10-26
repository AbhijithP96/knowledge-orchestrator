import re
import os
import numpy as np

from embeddings import embed_text_batch
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from ragas import metrics, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from dotenv import load_dotenv

load_dotenv()

# ==========================================
# RETRIEVAL METRICS
# ==========================================

class RetrievalMetrics:

    def _get_relevant_indices(self, query: str, retrieved_texts: List[str], thresh: float =0.5):

        query_embed = embed_text_batch([query])
        doc_embed = embed_text_batch(retrieved_texts)

        cosine_similarity_values = cosine_similarity(query_embed, doc_embed)[0]

        relevant_indices = set(
            idx for idx, score in enumerate(cosine_similarity_values) if score >= thresh
        )

        return relevant_indices


    def calculate_recall_at_k(self, query: str, retrieved_texts: List[str], top_k: int=3):

        if not retrieved_texts:
            return 0.0
        
        top_k_texts = retrieved_texts[:top_k]
        relevant_top_k = self._get_relevant_indices(query, top_k_texts)

        all_relevant = self._get_relevant_indices(query, retrieved_texts)
        total_relevant = len(all_relevant)

        if total_relevant == 0:
            return 0.0
        
        return len(relevant_top_k)/total_relevant

    def calculate_redundancy_rate(self, retrieved_texts: List[str], thresh:float = 0.9):
        """ Lower is Better """

        if len(retrieved_texts) < 2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer()
            tfid_matrix = vectorizer(retrieved_texts)

            similarity_matirx = cosine_similarity(tfid_matrix)

            n=len(retrieved_texts)
            upper_triangle = np.triu_indices(n, k=1)
            similarities = similarity_matirx[upper_triangle]

            redundant_pairs = np.sum(similarities >= thresh)
            total_pairs = len(similarities)

            return redundant_pairs/total_pairs
        
        except:
            return 0.0

    def calculate_avg_precsion(self, query: str, retrieved_texts: List[str]):

        if not retrieved_texts:
            return 0.0
        
        relevant_indices = sorted(self._get_relevant_indices(query, retrieved_texts))

        if not relevant_indices:
            return 0.0
        
        precisions = []

        for rank,_ in enumerate(retrieved_texts):
            if rank in relevant_indices:

                num_docs_so_far = sum(1 for r in relevant_indices if r<=rank)
                precisions.append(num_docs_so_far/ (rank +1))

        return sum(precisions)/len(relevant_indices)


# ==========================================
# GENERATOR METRICS
# ==========================================

class GeneratorMetrics:

    def __init__(self):
        
        self.embedder = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{os.environ['EMBED_MODEL']}")
        self.llm = Ollama(model=os.environ['OLLAMA_MODEL'])

        self.embedder = LangchainEmbeddingsWrapper(self.embedder)
        self.llm = LangchainLLMWrapper(self.llm)


    def evaluate_with_ragas(self, query: str, answer: str, context: List[str], gt: str = None):

        try:
            eval_dataset = Dataset.from_dict({
                'question' : [query],
                'answer' : [answer],
                'retrieved_contexts' : [context],
                'reference': [gt] if gt else [""]
            })

            eval_metrics = [
                metrics.faithfulness,
                metrics.answer_relevancy,
            ]

            if gt:
                eval_metrics.append(metrics.context_precision)
                eval_metrics.append(metrics.context_recall)

            result = evaluate(
                dataset=eval_dataset,
                metrics=eval_metrics,
                llm=self.llm,
                embeddings=self.embedder,
                raise_exceptions=False
            )

            scores = {}

            for metric in eval_metrics:
                metric_name = metric.name
                scores[metric_name] = float(result[metric_name][0])

            return scores

        except Exception as e:
            print('Error Encounterd during Ragas Evalauation')
            print(str(e))
            
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0
            }



if __name__ == "__main__":
    evaluator = GeneratorMetrics()

    query = "What is the capital of France?"
    context = "France is a country in Europe. Its capital city is Paris, known for the Eiffel Tower."
    answer = "The capital of France is Paris."
    ground_truth = "Paris"

    evaluator.evaluate_with_ragas(query, answer, context, ground_truth)

    # just an example, nothing to see