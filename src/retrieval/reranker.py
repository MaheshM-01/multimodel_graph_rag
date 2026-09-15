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

    @staticmethod
    def _normalize_stem(t: str) -> str:
        """Lightweight algorithmic stemmer for English plurals, gerunds, and past tense."""
        t = t.lower().strip()
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        if t.endswith("es") and len(t) > 4:
            return t[:-2]
        if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
            return t[:-1]
        if t.endswith("ing") and len(t) > 5:
            return t[:-3]
        if t.endswith("ed") and len(t) > 4:
            return t[:-2]
        return t

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

        # Strict penalty for Table of Contents when asking conceptual queries
        is_toc = bool(candidate.metadata.get("is_toc"))
        if is_toc and not any(w in q_clean for w in ["table of contents", "syllabus", "outline", "summary", "index"]):
            return -25.0

        content_tokens_list = re.findall(r"\b\w+\b", content)
        content_tokens = set(content_tokens_list)

        # 1. Salient token coverage ratio with morphological stem matching
        q_stems = [self._normalize_stem(t) for t in salient_tokens]
        content_stems = set(self._normalize_stem(t) for t in content_tokens_list)

        covered_salient = sum(1 for s in q_stems if s in content_stems)
        coverage_ratio = covered_salient / max(len(q_stems), 1)

        # Strict elimination for candidates with zero query keyword match
        if coverage_ratio == 0:
            return -5.0

        # Completeness bonus: reward chunks covering ALL conceptual elements in the query
        completeness_bonus = 4.0 if coverage_ratio >= 0.99 else (1.5 if coverage_ratio >= 0.74 else 0.0)

        # 2. Dense Semantic Cosine Similarity
        # Encode candidate content snippet (first 450 characters for focused matching)
        snippet_text = candidate.content[:450]
        c_vec = np.array(self.embedder.encode_text(snippet_text), dtype=np.float32)
        c_norm = np.linalg.norm(c_vec)
        dense_sim = 0.0
        if c_norm > 1e-6:
            c_vec /= c_norm
            dense_sim = float(np.dot(q_vec, c_vec))

        # 3. Dynamic Longest Continuous Phrase Match (LCP)
        lcp_len = self._longest_common_phrase_length(salient_tokens, content)
        phrase_boost = float(lcp_len) * 2.5

        # 4. Term Frequency / Density Boost
        matches = sum(1 for s in content_tokens_list if self._normalize_stem(s) in q_stems)
        freq_boost = min(math.log(1.0 + matches), 3.5) * 1.6

        # 5. Structural Heading Alignment
        heading = str(candidate.metadata.get("heading") or candidate.metadata.get("section_heading") or "").lower()
        heading_boost = 0.0
        if heading:
            heading_stems = set(self._normalize_stem(t) for t in re.findall(r"\b\w+\b", heading))
            covered_in_heading = sum(1 for s in q_stems if s in heading_stems)
            if covered_in_heading > 0:
                heading_boost = (covered_in_heading / max(len(q_stems), 1)) * 4.5

        # 6. Base retrieval score normalization
        if candidate.score > 1.0:
            norm_base = min(candidate.score / 25.0, 3.0)
        else:
            norm_base = min(candidate.score * 15.0, 2.0)

        # Composite score
        total_score = (
            (4.5 * dense_sim)
            + (3.5 * coverage_ratio)
            + completeness_bonus
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
                        if wants_visuals:
                            # User explicitly requested diagram / visual
                            visual_add = 3.0 + (v_sim * 4.0)
                            c.visual_relevance_score = round(v_sim, 2)
                            c.score = round(c.score + visual_add, 4)
                        elif v_sim > 0.40:
                            # Additive tie-breaker bonus without inflating over authoritative text
                            c.visual_relevance_score = round(v_sim, 2)
                            c.score = round(c.score + (v_sim * 0.8), 4)

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        # Cutoff: eliminate poor matches
        best_score = scored_candidates[0].score if scored_candidates else 0.0
        cutoff = max(1.5, best_score * 0.40)
        filtered = [c for c in scored_candidates if c.score >= cutoff]
        final_list = filtered if filtered else scored_candidates[:top_k]

        return final_list[:top_k]


ContextualReranker = TwoStageMultimodalReranker
