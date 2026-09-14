"""Domain models for Generation, Multimodal Citations, and Grounding."""

from typing import Any
from pydantic import BaseModel, Field
from src.domain.graph import SubGraph
from src.domain.multimodal import BoundingBox


class MultimodalCitation(BaseModel):
    """Citation linking generated answer assertions to source multimodal context."""

    citation_index: int
    source_chunk_id: str
    document_id: str
    page_number: int | None = None
    media_url: str | None = None
    bounding_box: BoundingBox | None = None
    snippet: str
    modality: str


class GroundingMetadata(BaseModel):
    """Provenance and grounding details for explainability."""

    retrieved_chunk_count: int
    graph_entities_used: list[str] = Field(default_factory=list)
    graph_relationships_used: list[str] = Field(default_factory=list)
    subgraph: SubGraph | None = None
    communities_consulted: list[str] = Field(default_factory=list)


class RAGRequest(BaseModel):
    """Client request for RAG synthesis."""

    query: str
    document_name: str | None = None
    model: str | None = None
    rag_mode: str | None = None
    image_url_or_path: str | None = None
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 1500
    stream: bool = False


class RAGResponse(BaseModel):
    """Complete RAG synthesized response with multimodal grounding."""

    query: str
    answer: str
    citations: list[MultimodalCitation] = Field(default_factory=list)
    grounding: GroundingMetadata
    latency_seconds: float = 0.0
