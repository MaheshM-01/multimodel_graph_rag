"""Comprehensive Basic and Advanced Retrieval & RAGAS Evaluation Suite.

Evaluates:
- Basic IR Metrics: MRR, Hit Rate@K, Precision@K, MAP@K, NDCG@K
- Advanced RAGAS Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevance
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Dict


@dataclass
class EvalTestCase:
    query: str
    expected_pages: List[int]
    expected_concepts: List[str]
    description: str
    category: str


# ---------------------------------------------------------------------------
# 1. Basic Information Retrieval (IR) Metrics
# ---------------------------------------------------------------------------

def compute_reciprocal_rank(retrieved_pages: List[int], expected_pages: List[int]) -> float:
    """Computes Reciprocal Rank (1 / rank of the first relevant item)."""
    for idx, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            return 1.0 / idx
    return 0.0


def compute_hit_rate(retrieved_pages: List[int], expected_pages: List[int], k: int = 5) -> float:
    """Computes Hit Rate @ K (1 if any relevant item is found in top K, else 0)."""
    top_k = retrieved_pages[:k]
    return 1.0 if any(p in expected_pages for p in top_k) else 0.0


def compute_precision_at_k(retrieved_pages: List[int], expected_pages: List[int], k: int = 5) -> float:
    """Computes Precision @ K (R-Precision adjusted when ground truth count is less than K)."""
    if not retrieved_pages or not expected_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    hits = sum(1 for p in top_k if p in expected_pages)
    denom = min(len(top_k), max(1, len(expected_pages)))
    return min(1.0, hits / denom)


def compute_map_at_k(retrieved_pages: List[int], expected_pages: List[int], k: int = 5) -> float:
    """Computes Mean Average Precision @ K."""
    if not expected_pages or not retrieved_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    score = 0.0
    num_hits = 0
    for idx, p in enumerate(top_k, start=1):
        if p in expected_pages:
            num_hits += 1
            score += num_hits / idx
    denom = min(len(expected_pages), len(top_k), k)
    return min(1.0, score / max(1, denom))


def compute_ndcg_at_k(retrieved_pages: List[int], expected_pages: List[int], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain @ K (NDCG@K)."""
    if not expected_pages or not retrieved_pages:
        return 0.0
    top_k = retrieved_pages[:k]

    dcg = 0.0
    for idx, p in enumerate(top_k, start=1):
        rel = 1.0 if p in expected_pages else 0.0
        dcg += (2.0 ** rel - 1.0) / math.log2(idx + 1.0)

    # Ideal DCG
    idcg = 0.0
    ideal_hits = min(len(expected_pages), len(top_k), k)
    for idx in range(1, ideal_hits + 1):
        idcg += (2.0 ** 1.0 - 1.0) / math.log2(idx + 1.0)

    return min(1.0, dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# 2. Advanced RAGAS Metrics
# ---------------------------------------------------------------------------

def compute_ragas_context_precision(retrieved_pages: List[int], expected_pages: List[int], k: int = 5) -> float:
    """Computes RAGAS Context Precision @ K:
    Measures the signal-to-noise ratio of retrieved contexts by calculating
    the weighted precision of relevant items across the ranked list.
    """
    if not retrieved_pages or not expected_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    score = 0.0
    num_hits = 0
    for idx, p in enumerate(top_k, start=1):
        if p in expected_pages:
            num_hits += 1
            score += (num_hits / idx)
    if num_hits == 0:
        return 0.0
    return score / num_hits


def compute_ragas_context_recall(retrieved_text: str, expected_concepts: List[str]) -> float:
    """Computes RAGAS Context Recall based on ground truth coverage."""
    if not expected_concepts:
        return 1.0
    text_lower = retrieved_text.lower()
    matches = sum(1 for c in expected_concepts if c.lower() in text_lower)
    return matches / len(expected_concepts)


def compute_ragas_faithfulness(generated_text: str, retrieved_text: str) -> float:
    """Computes RAGAS Faithfulness: measures grounded claims against context."""
    if not generated_text or not retrieved_text:
        return 0.0

    STOPWORDS = {
        "based", "provided", "according", "context", "following", "summary",
        "overview", "details", "which", "their", "there", "where", "these",
        "those", "about", "under", "using", "would", "could", "should",
        "with", "from", "into", "that", "this", "then", "than", "when",
        "what", "also", "have", "been", "each", "such", "more", "some",
        "specifically", "associated", "characterized", "critical", "profile",
        "between", "through", "while", "during", "across", "other", "both",
        "allows", "enables", "helps", "used", "uses", "described", "note",
        "first", "second", "third", "step", "steps", "system", "process"
    }

    # Split into clean, atomic sentence propositions
    raw_sentences = re.split(r"(?<=[.!?\n])\s+", generated_text)
    claim_sentences = []
    for s in raw_sentences:
        clean = s.strip()
        if not clean or clean.startswith(("#", "---", "===")) or (clean.startswith("[") and "]" in clean and len(clean) < 120):
            continue
        clean = re.sub(r"[*_`#\[\]\(\)]", " ", clean).strip()
        clean = re.sub(r"^\d+\.\s*", "", clean).strip()
        if len(clean) > 15 and not clean.endswith(":"):
            claim_sentences.append(clean)

    if not claim_sentences:
        return 1.0

    grounded = 0
    retrieved_lower = retrieved_text.lower()
    for s in claim_sentences:
        words = [w.lower() for w in re.findall(r"\b\w{3,}\b", s) if w.lower() not in STOPWORDS]
        if not words:
            continue
        anchored = sum(1 for w in words if w in retrieved_lower)
        if (anchored / len(words)) >= 0.15:
            grounded += 1

    return grounded / len(claim_sentences)


def compute_ragas_answer_relevance(query: str, generated_text: str, expected_concepts: List[str]) -> float:
    """Computes RAGAS Answer Relevance: alignment with core question intent."""
    text_lower = generated_text.lower()
    covered = sum(1 for c in expected_concepts if c.lower() in text_lower)
    return min(1.0, covered / max(1, len(expected_concepts) * 0.70))
