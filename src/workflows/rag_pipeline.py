"""End-to-end Multimodal Graph RAG pipeline with Smart Routing, Weighted RRF, and 2-Stage Reranking."""

from src.core.logging import logger
from src.core.telemetry import trace_action
from src.domain.generation import RAGRequest, RAGResponse
from src.domain.retrieval import QueryRequest
from src.generation.synthesizer import MultimodalSynthesizer
from src.retrieval.hybrid_fusion import HybridRetriever
from src.retrieval.query_router import QueryRouter
from src.retrieval.reranker import TwoStageMultimodalReranker


class RAGWorkflow:
    """DAG Pipeline: Query ➔ Smart Router ➔ Activated Channels Parallel Search ➔ Weighted RRF ➔ 2-Stage Rerank ➔ Synthesize."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: TwoStageMultimodalReranker,
        synthesizer: MultimodalSynthesizer,
        router: QueryRouter | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.synthesizer = synthesizer
        self.router = router or QueryRouter()

    @trace_action("rag_query_pipeline")
    async def execute(self, request: RAGRequest) -> RAGResponse:
        logger.info(f"[RAGWorkflow] Initiating Quad-Hybrid pipeline for query: '{request.query}'")

        # 1. Smart Query Routing (Intent & Activated Channels)
        query_req = QueryRequest(
            query_text=request.query,
            query_image_url_or_path=request.image_url_or_path,
            top_k=20,
        )
        decision = await self.router.route(query_req)
        logger.info(
            f"[RAGWorkflow:Router] Intent='{decision.intent.value}' | Active Channels={decision.active_channels} | Weights={decision.channel_weights}"
        )

        # 2. Parallel Search across Activated Channels with Weighted RRF
        candidates = await self.retriever.retrieve(
            request=query_req,
            active_channels=decision.active_channels,
            channel_weights=decision.channel_weights,
        )

        # 3. 2-Stage Multimodal Reranking (Text Cross-Encoder + Visual Grounding)
        reranked_items = await self.reranker.rerank(
            query=request.query,
            candidates=candidates,
            top_k=8,
        )

        # 4. Structured Context Assembly & Answer Synthesis
        response = await self.synthesizer.synthesize(request, reranked_items)
        return response
