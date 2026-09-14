"""Unit tests for Reciprocal Rank Fusion (RRF) algorithm."""

from src.core.constants import ModalityType
from src.domain.retrieval import SearchResult
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_logic():
    vector_list = [
        SearchResult(
            id="doc_1",
            modality=ModalityType.TEXT,
            score=0.95,
            content="Vector content 1",
            source_type="vector",
        ),
        SearchResult(
            id="doc_2",
            modality=ModalityType.TEXT,
            score=0.85,
            content="Vector content 2",
            source_type="vector",
        ),
    ]

    graph_list = [
        SearchResult(
            id="doc_2",
            modality=ModalityType.TEXT,
            score=1.0,
            content="Vector content 2",
            source_type="graph",
        ),
        SearchResult(
            id="doc_3",
            modality=ModalityType.IMAGE,
            score=1.0,
            content="Graph content 3",
            source_type="graph",
        ),
    ]

    ranked_lists = {
        "vector": vector_list,
        "graph": graph_list,
    }

    fused = reciprocal_rank_fusion(ranked_lists, k=60)

    # doc_2 appears in both lists (rank 2 in vector, rank 1 in graph)
    # doc_2 score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # doc_1 score = 1/(60+1) = 0.016393
    # doc_3 score = 1/(60+2) = 0.016129
    # doc_2 should be ranked #1
    assert len(fused) == 3
    assert fused[0].id == "doc_2"
    assert "vector" in fused[0].channel_ranks
    assert "graph" in fused[0].channel_ranks
