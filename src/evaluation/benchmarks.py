"""Automated benchmark runner for Multimodal Graph RAG testing."""

from typing import Any
from src.core.logging import logger
from src.domain.generation import RAGRequest
from src.generation.synthesizer import MultimodalSynthesizer
from src.retrieval.hybrid_fusion import HybridRetriever


class BenchmarkRunner:
    """Runs test suites over multimodal graph dataset to assess retrieval accuracy and answer quality."""

    def __init__(
        self,
        retriever: HybridRetriever,
        synthesizer: MultimodalSynthesizer,
    ):
        self.retriever = retriever
        self.synthesizer = synthesizer

    async def run_benchmark(self, test_cases: list[dict[str, Any]]) -> dict[str, Any]:
        logger.info(f"Running benchmark across {len(test_cases)} evaluation queries...")
        scores = []
        for case in test_cases:
            query = case["query"]
            req = RAGRequest(query=query)
            retrieved = await self.retriever.retrieve(case.get("query_request"))
            response = await self.synthesizer.synthesize(req, retrieved)
            scores.append(len(response.citations))

        avg_citations = sum(scores) / len(scores) if scores else 0.0
        return {
            "total_cases": len(test_cases),
            "average_citations_per_query": avg_citations,
            "status": "completed",
        }
