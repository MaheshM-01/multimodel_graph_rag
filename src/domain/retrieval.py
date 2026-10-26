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


class DomainType(str, Enum):
    """Domain index taxonomy for Mixture of Domain Indices (MoDI).

    Prevents noise, cross-domain false positives, and semantic drift across millions of documents.
    """

    ENGINEERING_TECHNICAL = "engineering_technical"   # AI, Deep Learning, Transformers, Algorithms, Code
    HARDWARE_SCHEMATICS = "hardware_schematics"       # Semiconductors, ICs, Pinouts, Schematics, PCB
    FINANCIAL_REGULATORY = "financial_regulatory"     # Balance sheets, SEC filings, Revenue, GAAP
    LEGAL_COMPLIANCE = "legal_compliance"             # Contracts, NDAs, Compliance, Patents, IP
    CORPORATE_GOVERNANCE = "corporate_governance"     # Supply chain, Board of Directors, Ownership, Vendor
    MEDICAL_BIOMEDICAL = "medical_biomedical"         # Clinical trials, Healthcare, BioBERT research
    CROSS_DOMAIN_ALL = "cross_domain_all"             # Global federated index fallback


class DomainScore(BaseModel):
    """Individual domain similarity and confidence score."""

    domain: DomainType
    similarity: float
    confidence: float
    label: str


class DomainRoutingResult(BaseModel):
    """Output from the Query Domain Embedding Router selecting specialized knowledge indices."""

    primary_domain: DomainType
    secondary_domain: DomainType | None = None
    scores: list[DomainScore] = Field(default_factory=list)
    search_space_pruned_pct: float = Field(
        default=0.0,
        description="Percentage of global knowledge corpus pruned from search space (e.g. 83.3%)",
    )
    active_collections: list[str] = Field(
        default_factory=list,
        description="Target Qdrant/index collections narrowed down for retrieval",
    )
    reasoning: str = ""


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
    domain_routing: DomainRoutingResult | None = None


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
    target_domain: str | None = None
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
