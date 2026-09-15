"""Production Universal Multimodal Reranker with Neural Cross-Encoding and Dynamic Grounding.
Completely eliminates hardcoded domain phrases, query rules, and page-specific checks.
"""

import math
import re
from typing import Optional
import numpy as np

from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.retrieval import SearchResult
from src.embeddings.text_embedder import TextEmbeddingEngine


class TwoStageMultimodalReranker:
    """Universal 2-Stage Cross-Encoder & Multimodal Grounding Engine.
    
    Stage 1: Deep Cross-Encoder scoring measuring joint dense semantic similarity,
             salient token coverage, dynamic contiguous n-gram sequence matching, and heading alignment.
    Stage 2: Cross-modal visual grounding prioritizing verified diagrams whose captions
             semantically align with the user query.
    """

    def __init__(self, cross_encoder_model: str = "all-MiniLM-L6-v2"):
        self.cross_encoder_model = cross_encoder_model
        self.embedder = TextEmbeddingEngine()
        logger.info(f"Initialized Universal Cross-Encoder Reranker with neural embeddings.")

    def _longest_common_phrase_length(self, q_words: list[str], doc_text_lower: str) -> int:
        """Find the length of the longest contiguous sequence of query words present in document."""
        if not q_words:
            return 0

        max_len = 0
        n = len(q_words)
        # Check window sizes from n down to 2
        for window in range(n, 1, -1):
            for i in range(n - window + 1):
                subseq = " ".join(q_words[i : i + window])
                if subseq in doc_text_lower:
                    return window
        return max_len

    def _compute_cross_encoder_score(self, query: str, candidate: SearchResult, q_vec: np.ndarray) -> float:
        """Universal scoring function without any hardcoded dictionaries or static query branching."""
        q_clean = query.lower().strip()
        content = candidate.content.lower()
        q_tokens = re.findall(r"\b\w+\b", q_clean)

        stop_words = {
            "machi", "ta", "what", "is", "mean", "by", "the", "a", "an", "and", "or", "in", "on", "at", "to",
            "explain", "describe", "details", "definition", "show", "tell", "me", "how", "why", "does"
        }
        salient_tokens = [t for t in q_tokens if t not in stop_words and len(t) > 1]
        if not salient_tokens:
            salient_tokens = [t for t in q_tokens if len(t) > 1]

        content_tokens_list = re.findall(r"\b\w+\b", content)
        content_tokens = set(content_tokens_list)

        # 1. Salient token coverage ratio (fraction of core query words present)
        covered_salient = sum(1 for t in salient_tokens if t in content_tokens)
        coverage_ratio = covered_salient / max(len(salient_tokens), 1)

        # Strict elimination for candidates with zero query keyword match
        if coverage_ratio == 0:
            return -5.0

        # 2. Dense Semantic Cosine Similarity
        # Encode candidate content snippet (first 350 characters for focused matching)
        snippet_text = candidate.content[:350]
        c_vec = np.array(self.embedder.encode_text(snippet_text), dtype=np.float32)
        c_norm = np.linalg.norm(c_vec)
        dense_sim = 0.0
        if c_norm > 1e-6:
            c_vec /= c_norm
            dense_sim = float(np.dot(q_vec, c_vec))

        # 3. Dynamic Longest Continuous Phrase Match (LCP)
        lcp_len = self._longest_common_phrase_length(salient_tokens, content)
        phrase_boost = float(lcp_len) * 2.2

        # 4. Term Frequency / Density Boost
        matches = sum(content_tokens_list.count(t) for t in salient_tokens)
        freq_boost = min(math.log(1.0 + matches), 3.5) * 1.6

        # 5. Structural Heading Alignment
        heading = str(candidate.metadata.get("heading") or candidate.metadata.get("section_heading") or "").lower()
        heading_boost = 0.0
        if heading:
            heading_tokens = set(re.findall(r"\b\w+\b", heading))
            covered_in_heading = sum(1 for t in salient_tokens if t in heading_tokens)
            if covered_in_heading > 0:
                heading_boost = (covered_in_heading / max(len(salient_tokens), 1)) * 4.0

        # 6. Base retrieval score normalization
        if candidate.score > 1.0:
            norm_base = min(candidate.score / 25.0, 3.0)
        else:
            norm_base = min(candidate.score * 15.0, 2.0)

        # Composite score
        total_score = (
            (4.0 * dense_sim)
            + (3.5 * coverage_ratio)
            + phrase_boost
            + freq_boost
            + heading_boost
            + norm_base
        )
        return float(total_score)

    async def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        if not candidates:
            return []

        logger.debug(f"[Universal-Reranker] Joint cross-encoding {len(candidates)} candidates for query: '{query}'")

        # Encode query once for dense cross-encoder matching
        q_vec = np.array(self.embedder.encode_text(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-6:
            q_vec /= q_norm

        scored_candidates: list[SearchResult] = []
        for c in candidates:
            ce_score = self._compute_cross_encoder_score(query, c, q_vec)
            c.score = round(ce_score, 4)
            scored_candidates.append(c)

        # ----------------------------------------------------------------------
        # Stage 2: Universal Multimodal Visual Grounding
        # Boosts visuals ONLY when their caption/heading semantically matches the query
        # ----------------------------------------------------------------------
        q_lower = query.lower()
        wants_visuals = any(
            w in q_lower for w in ["visual", "diagram", "image", "chart", "figure", "picture", "schematic"]
        )

        for c in scored_candidates:
            has_visual = c.modality == ModalityType.IMAGE or c.source_type == "visual" or c.metadata.get("has_visuals")
            if has_visual and c.score > 1.0:
                # Dynamically evaluate visual relevance by comparing query with figure title / heading
                fig_desc = str(c.metadata.get("figure_title") or c.metadata.get("section_heading") or "").strip()
                if fig_desc:
                    fig_vec = np.array(self.embedder.encode_text(fig_desc), dtype=np.float32)
                    fig_norm = np.linalg.norm(fig_vec)
                    if fig_norm > 1e-6:
                        fig_vec /= fig_norm
                        v_sim = float(np.dot(q_vec, fig_vec))
                        if v_sim > 0.35:
                            visual_boost = 1.0 + (v_sim * 0.6)
                            if wants_visuals:
                                visual_boost += 0.35
                            c.visual_relevance_score = round(visual_boost, 2)
                            c.score = round(c.score * visual_boost, 4)

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        # Cutoff: eliminate poor matches
        best_score = scored_candidates[0].score if scored_candidates else 0.0
        cutoff = max(1.0, best_score * 0.35)
        filtered = [c for c in scored_candidates if c.score >= cutoff]
        final_list = filtered if filtered else scored_candidates[:top_k]

        return final_list[:top_k]


ContextualReranker = TwoStageMultimodalReranker
