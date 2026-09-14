"""RAGAS Evaluation Module for Multimodal Graph RAG.
Evaluates Context Precision, Context Recall, Faithfulness, and Answer Relevance.
"""

import math
import re
from typing import Any, List
from src.core.logging import logger
from src.domain.generation import RAGRequest, RAGResponse
from src.domain.retrieval import SearchResult


def compute_context_precision(retrieved_pages: List[int], expected_pages: List[int], k: int = 5) -> float:
    """Computes Context Precision @ K (Mean Average Precision on document page rankings)."""
    if not expected_pages or not retrieved_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    relevant_count = 0
    precision_sum = 0.0
    for idx, page in enumerate(top_k):
        if page in expected_pages:
            relevant_count += 1
            precision_sum += relevant_count / (idx + 1)
    return precision_sum / min(len(expected_pages), k) if expected_pages else 0.0


def compute_context_recall(retrieved_text: str, expected_concepts: List[str]) -> float:
    """Computes Context Recall based on ground-truth domain concept coverage."""
    if not expected_concepts:
        return 1.0
    text_lower = retrieved_text.lower()
    matches = sum(1 for c in expected_concepts if c.lower() in text_lower)
    return matches / len(expected_concepts)


def compute_faithfulness(generated_text: str, retrieved_text: str) -> float:
    """Computes Faithfulness: measures grounded claims in generated answer against retrieved context."""
    sentences = [s.strip() for s in re.split(r"[.\n]", generated_text) if len(s.strip()) > 15]
    if not sentences:
        return 1.0
    grounded_count = 0
    retrieved_lower = retrieved_text.lower()
    for s in sentences:
        words = [w.lower() for w in re.findall(r"\b\w{4,}\b", s)]
        if not words:
            continue
        anchored = sum(1 for w in words if w in retrieved_lower)
        if (anchored / len(words)) >= 0.35:
            grounded_count += 1
    return grounded_count / len(sentences)


def compute_answer_relevance(query: str, generated_text: str, expected_concepts: List[str]) -> float:
    """Computes Answer Relevance: measures if synthesized response covers core query intent."""
    text_lower = generated_text.lower()
    covered = sum(1 for c in expected_concepts if c.lower() in text_lower)
    return min(1.0, covered / (len(expected_concepts) * 0.75))
