"""Unit tests for workflow orchestrators."""

import pytest
from src.domain.generation import RAGRequest
from src.generation.llm_client import LLMClient
from src.generation.synthesizer import MultimodalSynthesizer
from src.retrieval.hybrid_fusion import HybridRetriever
from src.retrieval.reranker import ContextualReranker
from src.workflows.rag_pipeline import RAGWorkflow
from src.api.dependencies import get_hybrid_retriever


@pytest.mark.asyncio
async def test_rag_workflow_orchestration():
    retriever = get_hybrid_retriever()
    reranker = ContextualReranker()
    synthesizer = MultimodalSynthesizer(llm_client=LLMClient())

    workflow = RAGWorkflow(
        retriever=retriever,
        reranker=reranker,
        synthesizer=synthesizer,
    )

    request = RAGRequest(query="Explain the multimodal graph schema")
    response = await workflow.execute(request)

    assert response is not None
    assert response.query == "Explain the multimodal graph schema"
    assert response.grounding is not None
