"""Unit tests for BM25 sparse keyword retriever."""

import pytest
from src.domain.retrieval import QueryRequest
from src.retrieval.sparse_search import BM25Retriever


@pytest.mark.asyncio
async def test_bm25_retriever():
    retriever = BM25Retriever()
    retriever.index_chunk("chunk_1", "Quarterly revenue increased by 35 percent in Cloud Services.")
    retriever.index_chunk("chunk_2", "Employee onboarding documentation and compliance guidelines.")

    request = QueryRequest(query_text="Cloud Services revenue")
    results = await retriever.retrieve(request)

    assert len(results) > 0
    assert results[0].id == "chunk_1"
    assert results[0].score > 0.0
