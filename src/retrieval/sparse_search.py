"""Sparse / Okapi BM25 keyword search retriever for exact entity, code, and token matches."""

import math
import re
from collections import defaultdict
from typing import Any
from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.retrieval import QueryRequest, SearchResult
from src.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Sparse keyword retriever executing Okapi BM25 scoring over tokenized document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        logger.info("Initialized BM25Retriever for sparse keyword search")
        self._corpus: dict[str, dict[str, Any]] = {}
        self.k1 = k1
        self.b = b
        self._idf_cache: dict[str, float] = {}
        self._avg_dl: float = 1.0
        self._dirty = False

    def index_chunk(self, chunk_id: str, content: str, metadata: dict | None = None) -> None:
        """Add text chunk to BM25 in-memory index."""
        tokens = re.findall(r"\b\w+\b", content.lower())
        term_freq = defaultdict(int)
        for t in tokens:
            term_freq[t] += 1

        self._corpus[chunk_id] = {
            "content": content,
            "tokens": tokens,
            "term_freq": term_freq,
            "doc_len": len(tokens),
            "metadata": metadata or {},
        }
        self._dirty = True

    def _ensure_corpus_loaded(self) -> None:
        """Auto-populate corpus from DocumentSearchEngine if in-memory index is empty."""
        if self._corpus:
            return

        try:
            from src.retrieval.document_search import get_document_search_engine
            engine = get_document_search_engine()
            pages = engine._ensure_documents_indexed()
            for p in pages:
                cid = f"{p['document_id']}_p{p['page_number']}"
                self.index_chunk(cid, p["text"], metadata=p)
            logger.info(f"[BM25Retriever] Auto-populated {len(self._corpus)} chunks from documents")
        except Exception as e:
            logger.warning(f"[BM25Retriever] Failed to auto-populate corpus: {e}")

    def _update_idf(self) -> None:
        if not self._dirty and self._idf_cache:
            return

        total_docs = len(self._corpus)
        if total_docs == 0:
            self._idf_cache = {}
            self._avg_dl = 1.0
            return

        total_tokens = sum(doc["doc_len"] for doc in self._corpus.values())
        self._avg_dl = max(1.0, total_tokens / total_docs)

        df = defaultdict(int)
        for doc in self._corpus.values():
            for term in set(doc["tokens"]):
                df[term] += 1

        self._idf_cache = {}
        for term, count in df.items():
            # Standard Okapi BM25 IDF
            self._idf_cache[term] = math.log((total_docs - count + 0.5) / (count + 0.5) + 1.0)

        self._dirty = False

    async def retrieve(self, request: QueryRequest) -> list[SearchResult]:
        if not request.query_text:
            return []

        self._ensure_corpus_loaded()
        self._update_idf()

        query = request.query_text.lower()
        query_terms = re.findall(r"\b\w+\b", query)
        stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "of", "to", "is", "are", "what", "explain"}
        filtered_terms = [t for t in query_terms if t not in stop_words and len(t) > 1]
        if not filtered_terms:
            filtered_terms = query_terms

        logger.debug(f"Executing Okapi BM25 sparse search for terms: {filtered_terms}")
        results: list[SearchResult] = []

        for chunk_id, data in self._corpus.items():
            score = 0.0
            doc_len = data["doc_len"]
            term_freq = data["term_freq"]

            for q in filtered_terms:
                if q in term_freq:
                    tf = term_freq[q]
                    idf = self._idf_cache.get(q, 1.0)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_dl))
                    score += idf * (numerator / denominator)

            # Substring exact boost
            text_lower = data["content"].lower()
            if request.query_text.lower() in text_lower:
                score += 8.0

            # Technical phrase boost
            for phr in ["transformer", "attention mechanism", "back propagation", "forward and backward", "deep neural network", "chain rule"]:
                if phr in query and phr in text_lower:
                    score += 5.0

            # Downweight Table of Contents / Disclaimer Cover pages (Page 1/2)
            page_num = data.get("metadata", {}).get("page_number", 999)
            if page_num <= 2:
                if any(toc_kw in text_lower for toc_kw in ["table of contents", "contents", "personal notes", "arpit singh", "linkedin"]):
                    score -= 25.0

            if score > 0.1:
                results.append(
                    SearchResult(
                        id=chunk_id,
                        modality=ModalityType.TEXT,
                        score=round(score, 4),
                        content=data["content"],
                        source_type="sparse_bm25",
                        metadata=data.get("metadata", {}),
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: request.top_k]
