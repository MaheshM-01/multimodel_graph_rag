"""Weighted Reciprocal Rank Fusion (Weighted RRF) across Quad-Hybrid Retrieval Channels."""

from collections import defaultdict
from src.core.logging import logger
from src.domain.retrieval import FusionItem, QueryRequest, SearchResult
from src.retrieval.base import BaseRetriever
from src.retrieval.graph_search import GraphRetriever
from src.retrieval.sparse_search import BM25Retriever
from src.retrieval.vector_search import VectorRetriever
from src.retrieval.visual_search import VisualRetriever


def weighted_reciprocal_rank_fusion(
    ranked_lists: dict[str, list[SearchResult]],
    channel_weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[FusionItem]:
    """Combines multiple ranked lists using Weighted Reciprocal Rank Fusion (Weighted RRF).

    Formula:
        RRF_Score(d) = SUM_{c in Channels} [ w_c * (1 / (k + rank_c(d))) ]

    Weights default to:
        dense: 0.35, graph: 0.35, sparse: 0.15, visual: 0.15
    """
    weights = channel_weights or {
        "dense": 0.35,
        "graph": 0.35,
        "sparse": 0.15,
        "visual": 0.15,
    }

    scores: dict[str, float] = defaultdict(float)
    items_by_id: dict[str, SearchResult] = {}
    channel_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    applied_weights: dict[str, dict[str, float]] = defaultdict(dict)

    for channel_name, results in ranked_lists.items():
        w = weights.get(channel_name, 1.0)
        for rank, item in enumerate(results, start=1):
            rrf_term = w * (1.0 / (k + rank))
            scores[item.id] += rrf_term
            items_by_id[item.id] = item
            channel_ranks[item.id][channel_name] = rank
            applied_weights[item.id][channel_name] = w

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_items: list[FusionItem] = []
    for item_id in sorted_ids:
        original = items_by_id[item_id]
        fused_items.append(
            FusionItem(
                id=original.id,
                modality=original.modality,
                content=original.content,
                image_url=original.image_url,
                rrf_score=round(scores[item_id], 6),
                channel_ranks=channel_ranks[item_id],
                channel_weights_applied=applied_weights[item_id],
                data_points=original.data_points,
                metadata=original.metadata,
            )
        )

    return fused_items


# Backwards compatibility alias
reciprocal_rank_fusion = weighted_reciprocal_rank_fusion


class HybridRetriever(BaseRetriever):
    """Quad-Hybrid Retriever coordinating Dense, Sparse, Visual, and Graph channels with Weighted RRF."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        sparse_retriever: BM25Retriever | None = None,
        visual_retriever: VisualRetriever | None = None,
        rrf_k: int = 60,
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.visual_retriever = visual_retriever
        self.rrf_k = rrf_k
        logger.info("Initialized Quad-HybridRetriever with Weighted RRF")

    async def retrieve(
        self,
        request: QueryRequest,
        active_channels: list[str] | None = None,
        channel_weights: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        logger.info(f"Executing Quad-Hybrid retrieval for: '{request.query_text}'")

        channels = active_channels or ["dense", "sparse", "visual", "graph"]
        ranked_lists: dict[str, list[SearchResult]] = {}

        # 1. Dense Semantic Search
        if "dense" in channels:
            ranked_lists["dense"] = await self.vector_retriever.retrieve(request)

        # 2. Sparse BM25 Search
        if "sparse" in channels:
            ranked_lists["sparse"] = await self.sparse_retriever.retrieve(request)

        # 3. Knowledge Graph Traversal
        if "graph" in channels:
            ranked_lists["graph"] = await self.graph_retriever.retrieve(request)

        # 4. Multimodal / Visual Search
        if "visual" in channels and self.visual_retriever:
            ranked_lists["visual"] = await self.visual_retriever.retrieve(request)

        # 5. Execute Weighted RRF
        weights = channel_weights or request.channel_weights
        fused = weighted_reciprocal_rank_fusion(ranked_lists, channel_weights=weights, k=self.rrf_k)

        return [
            SearchResult(
                id=item.id,
                modality=item.modality,
                score=item.rrf_score,
                content=item.content,
                image_url=item.image_url,
                source_type="quad_hybrid_fusion",
                data_points=item.data_points,
                metadata={
                    "channel_ranks": item.channel_ranks,
                    "channel_weights": item.channel_weights_applied,
                    **item.metadata,
                },
            )
            for item in fused[: request.top_k]
        ]
