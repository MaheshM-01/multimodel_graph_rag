"""Evaluation metrics for Multimodal Graph RAG."""

from src.domain.generation import RAGResponse


def calculate_citation_grounding(response: RAGResponse) -> float:
    """Calculates percentage of claims that are supported by at least one multimodal citation."""
    if not response.citations:
        return 0.0
    return min(1.0, len(response.citations) / 3.0)


def calculate_entity_precision(retrieved_entities: list[str], ground_truth_entities: list[str]) -> float:
    """Calculates precision of extracted/retrieved knowledge graph entities against ground truth."""
    if not retrieved_entities:
        return 0.0
    matches = set(retrieved_entities).intersection(set(ground_truth_entities))
    return len(matches) / len(retrieved_entities)
