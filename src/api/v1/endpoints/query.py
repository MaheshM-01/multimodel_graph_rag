"""Multimodal Graph RAG query endpoint with Query Routing, Quad-Hybrid Retrieval, and 2-Stage Reranking."""

from fastapi import APIRouter, Depends
from src.api.dependencies import (
    get_hybrid_retriever,
    get_query_router,
    get_reranker,
    get_synthesizer,
)
from src.core.logging import logger
from src.domain.generation import RAGRequest, RAGResponse
from src.domain.retrieval import QueryRequest
from src.generation.synthesizer import MultimodalSynthesizer
from src.retrieval.document_search import get_document_search_engine
from src.retrieval.hybrid_fusion import HybridRetriever
from src.retrieval.query_router import QueryRouter
from src.retrieval.reranker import TwoStageMultimodalReranker

router = APIRouter()


@router.post("/chat", response_model=RAGResponse, tags=["Query"])
async def query_rag(
    request: RAGRequest,
    router_agent: QueryRouter = Depends(get_query_router),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: TwoStageMultimodalReranker = Depends(get_reranker),
    synthesizer: MultimodalSynthesizer = Depends(get_synthesizer),
) -> RAGResponse:
    """Execute end-to-end Multimodal Graph RAG: Intent Routing -> Quad-Hybrid Fusion -> 2-Stage Reranking -> Synthesis."""
    logger.info(f"Incoming query: '{request.query}', document: '{request.document_name}'")

    query_req = QueryRequest(
        query_text=request.query,
        query_image_url_or_path=request.image_url_or_path,
        top_k=12,
    )

    # 1. Intent Routing & Dynamic Channel Weighting
    decision = await router_agent.route(query_req)
    logger.info(f"[QueryRouter] Intent: {decision.intent.value}, channels: {decision.active_channels}")

    # 2. Quad-Hybrid Retrieval (Dense, Sparse BM25, Visual, Graph) with Weighted RRF
    retrieved_items = await retriever.retrieve(
        query_req,
        active_channels=decision.active_channels,
        channel_weights=decision.channel_weights,
    )

    # 3. Document Search Engine integration for page-level evidence
    doc_search = get_document_search_engine()
    doc_results = await doc_search.search(
        query=request.query,
        document_name=request.document_name,
        top_k=6,
    )

    if doc_results:
        # Avoid duplicate IDs
        existing_ids = {it.id for it in retrieved_items}
        for dr in doc_results:
            if dr.id not in existing_ids:
                retrieved_items.append(dr)

    if not retrieved_items:
        logger.warning(f"No retrieval results found for query: '{request.query}'")

    # 4. 2-Stage Multimodal Reranking (Cross-Encoder + Visual Grounding)
    reranked_items = await reranker.rerank(request.query, retrieved_items, top_k=8)

    # 5. Multimodal Grounded Synthesis with Citations
    response = await synthesizer.synthesize(request, reranked_items)
    return response
