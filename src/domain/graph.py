"""Domain models for Knowledge Graph entities, edges, and subgraphs."""

from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field
from src.core.constants import EdgeType, NodeType


class BaseNode(BaseModel):
    """Base class for graph nodes."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    label: NodeType
    properties: dict[str, Any] = Field(default_factory=dict)


class EntityNode(BaseNode):
    """Semantic entity extracted from multimodal content (Person, Concept, Metric, etc.)."""

    label: NodeType = NodeType.ENTITY
    name: str
    entity_type: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)


class ChunkNode(BaseNode):
    """Represents a text or table chunk node inside the graph."""

    label: NodeType = NodeType.CHUNK
    chunk_id: str
    document_id: str
    content_snippet: str
    page_number: int | None = None


class ImageNode(BaseNode):
    """Visual asset node linked into the Knowledge Graph."""

    label: NodeType = NodeType.IMAGE
    media_asset_id: str
    image_url: str
    caption: str | None = None
    vlm_summary: str | None = None


class GraphEdge(BaseModel):
    """Directed edge connecting two nodes in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    relationship_type: str = EdgeType.RELATES_TO
    description: str | None = None
    weight: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)


class SubGraph(BaseModel):
    """Subgraph extracted for context reasoning."""

    nodes: list[BaseNode | EntityNode | ImageNode | ChunkNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class CommunityReport(BaseModel):
    """Hierarchical community summary report (GraphRAG style)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    community_id: int
    level: int
    title: str
    summary: str
    full_content: str
    findings: list[str] = Field(default_factory=list)
    rating: float = Field(default=5.0, description="Importance or salience score")
    member_node_ids: list[str] = Field(default_factory=list)
