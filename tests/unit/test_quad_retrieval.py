"""Unit tests verifying Quad-Hybrid Retrieval, Smart Router, Weighted RRF, and 2-Stage Reranking."""

import pytest
from src.core.constants import ModalityType
from src.domain.retrieval import QueryIntent, QueryRequest, SearchResult
from src.embeddings.vision_embedder import VisionEmbeddingEngine
from src.retrieval.hybrid_fusion import weighted_reciprocal_rank_fusion
from src.retrieval.query_router import QueryRouter
from src.retrieval.reranker import TwoStageMultimodalReranker
from src.retrieval.visual_search import VisualRetriever
from src.storage.vector_store import QdrantVectorStore


@pytest.mark.asyncio
async def test_query_router_intent_classification():
    router = QueryRouter()

    # 1. Visual Intent
    req_visual = QueryRequest(query_text="Show me the revenue trend chart for 2024")
    decision_visual = await router.route(req_visual)
    assert decision_visual.intent == QueryIntent.VISUAL_COMPARATIVE
    assert "visual" in decision_visual.active_channels
    assert decision_visual.channel_weights["visual"] >= 0.40

    # 2. Global Thematic Intent
    req_global = QueryRequest(query_text="Summarize the main themes of the annual report")
    decision_global = await router.route(req_global)
    assert decision_global.intent == QueryIntent.GLOBAL_COMMUNITY
    assert "graph" in decision_global.active_channels

    # 3. Analytical Intent
    req_analytical = QueryRequest(query_text="How many security incidents occurred in Q3?")
    decision_analytical = await router.route(req_analytical)
    assert decision_analytical.intent == QueryIntent.ANALYTICAL_CYPHER
    assert decision_analytical.suggested_cypher is not None

    # 4. Factual Entity Intent
    req_factual = QueryRequest(query_text="What are the specifications of Model X?")
    decision_factual = await router.route(req_factual)
    assert decision_factual.intent == QueryIntent.FACTUAL_ENTITY
    assert "Model X" in decision_factual.extracted_entities


def test_weighted_reciprocal_rank_fusion():
    dense_results = [
        SearchResult(
            id="doc_dense",
            modality=ModalityType.TEXT,
            score=0.9,
            content="Dense match",
            source_type="dense",
        )
    ]
    visual_results = [
        SearchResult(
            id="doc_visual",
            modality=ModalityType.IMAGE,
            score=0.8,
            content="Visual chart match",
            source_type="visual",
            image_url="s3://charts/rev.png",
        )
    ]

    ranked_lists = {
        "dense": dense_results,
        "visual": visual_results,
    }

    # Custom weights prioritizing visual channel
    weights = {"dense": 0.2, "visual": 0.8}
    fused = weighted_reciprocal_rank_fusion(ranked_lists, channel_weights=weights, k=60)

    assert len(fused) == 2
    # doc_visual has weight 0.8 * 1/61 = 0.01311
    # doc_dense has weight 0.2 * 1/61 = 0.00327
    # doc_visual must rank #1
    assert fused[0].id == "doc_visual"
    assert fused[0].channel_weights_applied["visual"] == 0.8


@pytest.mark.asyncio
async def test_two_stage_multimodal_reranker():
    reranker = TwoStageMultimodalReranker()
    candidates = [
        SearchResult(
            id="doc_text",
            modality=ModalityType.TEXT,
            score=0.015,
            content="General operating expenses in Cloud division.",
            source_type="dense",
        ),
        SearchResult(
            id="doc_chart",
            modality=ModalityType.IMAGE,
            score=0.014,
            content="Bar chart showing quarterly revenue growth percentage.",
            source_type="visual",
            data_points={"Q3_growth": "+35%"},
        ),
    ]

    # Query targeting visual growth
    query = "Show me the quarterly revenue chart"
    reranked = await reranker.rerank(query, candidates, top_k=2)

    assert len(reranked) == 2
    # Visual item should get boosted by Stage 2 Visual Grounding check
    assert reranked[0].id == "doc_chart"
    assert reranked[0].visual_relevance_score is not None
    assert reranked[0].visual_relevance_score > 1.0


@pytest.mark.asyncio
async def test_visual_retriever_query():
    vector_store = QdrantVectorStore()
    vision_embedder = VisionEmbeddingEngine()
    visual_retriever = VisualRetriever(vector_store, vision_embedder)

    req = QueryRequest(query_text="Architecture diagram")
    results = await visual_retriever.retrieve(req)

    # Scaffolding returns search results list
    assert isinstance(results, list)
