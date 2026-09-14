"""Domain models for Hybrid Search, Vector Retrieval, Graph Traversal, and Query Routing."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
from src.core.constants import ModalityType


class QueryIntent(str, Enum):
    """Classified user query intent determining active retrieval channels."""

    FACTUAL_ENTITY = "factual_entity"         # Target: Dense + Sparse + Local Graph K-hop
    GLOBAL_COMMUNITY = "global_community"     # Target: Global Leiden/Louvain Community Search + Dense
    VISUAL_COMPARATIVE = "visual_comparative" # Target: Multimodal (ColPali/SigLIP) + Sparse + Local Graph
    ANALYTICAL_CYPHER = "analytical_cypher"   # Target: Text2Cypher + Sparse
    GENERAL_HYBRID = "general_hybrid"         # Target: All 4 channels (Quad-Hybrid)


class RouterDecision(BaseModel):
    """Output from the Query Router orchestrating retrieval execution."""

    intent: QueryIntent
    active_channels: list[str] = Field(
        default_factory=lambda: ["dense", "sparse", "visual", "graph"],
        description="Active channels selected from: 'dense', 'sparse', 'visual', 'graph'",
    )
    channel_weights: dict[str, float] = Field(
        default_factory=lambda: {"dense": 0.35, "graph": 0.35, "sparse": 0.15, "visual": 0.15}
    )
    extracted_entities: list[str] = Field(default_factory=list)
    suggested_cypher: str | None = None
    reasoning: str | None = None


class FilterCriteria(BaseModel):
    """Metadata and modality filters for retrieval queries."""

    document_ids: list[str] | None = None
    modalities: list[ModalityType] | None = None
    entity_types: list[str] | None = None
    min_score: float = 0.0
    extra_filters: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Retrieval query request containing text and/or visual query inputs."""

    query_text: str | None = None
    query_image_url_or_path: str | None = None
    top_k: int = 10
    include_graph_subgraph: bool = True
    include_community_reports: bool = True
    channel_weights: dict[str, float] | None = None
    filters: FilterCriteria | None = None


class SearchResult(BaseModel):
    """Individual retrieved result item with provenance."""

    id: str
    modality: ModalityType
    score: float
    content: str
    image_url: str | None = None
    source_type: str = Field(..., description="'dense', 'sparse', 'visual', 'graph', 'community', or 'cypher'")
    visual_relevance_score: float | None = None
    data_points: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FusionItem(BaseModel):
    """Item merged across multiple retrieval channels using Weighted RRF."""

    id: str
    modality: ModalityType
    content: str
    image_url: str | None = None
    rrf_score: float = 0.0
    channel_ranks: dict[str, int] = Field(
        default_factory=dict,
        description="Rank of item in each channel, e.g. {'dense': 1, 'graph': 4}",
    )
    channel_weights_applied: dict[str, float] = Field(default_factory=dict)
    data_points: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
