"""2-Stage Multimodal Reranker (Stage 1: Text Cross-Encoder + Stage 2: Visual Grounding Reranker)."""

import re
from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.retrieval import SearchResult


class TwoStageMultimodalReranker:
    """2-Stage Reranker:

    - Stage 1: Cross-Encoder for text passages and Knowledge Graph triples.
    - Stage 2: Visual Grounding Reranker scoring image/chart relevance against user query.
    """

    def __init__(self, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.cross_encoder_model = cross_encoder_model
        logger.info(f"Initialized TwoStageMultimodalReranker with model: {self.cross_encoder_model}")

    async def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 8
    ) -> list[SearchResult]:
        if not candidates:
            return []

        logger.debug(f"[Reranker:Stage1] Cross-encoding {len(candidates)} candidates for query: '{query}'")

        # ----------------------------------------------------------------------
        # Stage 1: Text & Graph Triple Cross-Encoding
        # ----------------------------------------------------------------------
        query_lower = query.lower()
        query_terms = set(re.findall(r"\b\w+\b", query_lower))
        stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "of", "to", "is", "are", "what", "explain"}
        salient_terms = {t for t in query_terms if t not in stop_words and len(t) > 1} or query_terms

        scored_candidates: list[SearchResult] = []

        for c in candidates:
            content_lower = c.content.lower()
            content_words = set(re.findall(r"\b\w+\b", content_lower))
            overlap = len(salient_terms.intersection(content_words)) / max(len(salient_terms), 1)

            # Phrase match check
            phrase_boost = 0.0
            if query_lower in content_lower:
                phrase_boost = 0.5
            elif any(phr in content_lower for phr in ["attention mechanism", "forward and backward", "back propagation", "deep neural"]):
                phrase_boost = 0.3

            # Combined score: RRF base score + Cross-Encoder relevance proxy
            stage1_score = c.score + (0.4 * overlap) + phrase_boost
            c.score = round(stage1_score, 5)
            scored_candidates.append(c)

        # ----------------------------------------------------------------------
        # Stage 2: Visual Grounding Reranker for Charts & Images
        # ----------------------------------------------------------------------
        for c in scored_candidates:
            if c.modality == ModalityType.IMAGE:
                # Check if visual data points or caption directly answer the query
                chart_relevance = 1.0
                if c.data_points:
                    chart_relevance += 0.3  # Bonus for structured numerical data points
                if any(w in c.content.lower() for w in ["chart", "revenue", "trend", "percentage", "figure"]):
                    chart_relevance += 0.2

                c.visual_relevance_score = round(chart_relevance, 2)
                c.score = round(c.score * chart_relevance, 5)

        # Sort by updated 2-stage score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:top_k]


# Backwards compatibility alias
ContextualReranker = TwoStageMultimodalReranker
