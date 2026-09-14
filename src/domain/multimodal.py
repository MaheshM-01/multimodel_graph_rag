"""Domain models for Multimodal Documents, Chunks, and Media Assets."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field
from src.core.constants import ModalityType


class BoundingBox(BaseModel):
    """Normalized spatial bounding box (coordinates 0.0 to 1.0)."""

    x_min: float = Field(..., description="Top-left x coordinate")
    y_min: float = Field(..., description="Top-left y coordinate")
    x_max: float = Field(..., description="Bottom-right x coordinate")
    y_max: float = Field(..., description="Bottom-right y coordinate")
    page_number: int = Field(default=1, description="1-indexed document page number")


class VisualEntityLink(BaseModel):
    """Links a specific region in an image/chart to a Knowledge Graph Entity."""

    entity_id: str
    entity_name: str
    bounding_box: BoundingBox
    confidence: float = 1.0
    relation_type: str = "VISUALLY_REPRESENTS"


class MediaAsset(BaseModel):
    """Represents an uploaded or extracted media item (image, audio, chart)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    modality: ModalityType
    mime_type: str
    file_path_or_url: str
    size_bytes: int = 0
    width: int | None = None
    height: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseChunk(BaseModel):
    """Base class for all ingested document chunks."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    modality: ModalityType
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TextChunk(BaseChunk):
    """Text-based chunk extracted from a document."""

    modality: ModalityType = ModalityType.TEXT
    content: str
    token_count: int | None = None
    page_number: int | None = None
    section_title: str | None = None


class ParentChunk(BaseChunk):
    """Hierarchical Parent chunk (1000-1500 tokens) representing a full section for Graph & LLM reasoning."""

    modality: ModalityType = ModalityType.TEXT
    section_title: str
    content: str
    token_count: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    child_chunk_ids: list[str] = Field(default_factory=list)


class ChildChunk(BaseChunk):
    """Hierarchical Child chunk (200-300 tokens) indexed in Vector DB with small-to-big retrieval."""

    modality: ModalityType = ModalityType.TEXT
    parent_chunk_id: str
    content: str
    contextual_header: str | None = Field(
        default=None,
        description="Anthropic-style contextual header prepended to resolve pronouns and isolate context",
    )
    full_searchable_content: str | None = None
    token_count: int | None = None
    page_number: int | None = None


class PropositionChunk(BaseChunk):
    """Atomic factual statement extracted from dense text for 1:1 Knowledge Graph triple mapping."""

    modality: ModalityType = ModalityType.TEXT
    source_chunk_id: str
    proposition_statement: str
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    temporal_context: str | None = None


class ImageChunk(BaseChunk):
    """Visual chunk (chart, diagram, figure, slide, page screenshot)."""

    modality: ModalityType = ModalityType.IMAGE
    media_asset_id: str
    image_url: str
    caption: str | None = None
    extracted_text_ocr: str | None = None
    vlm_summary: str | None = None
    bounding_box: BoundingBox | None = None
    surrounding_text_before: str | None = Field(
        default=None, description="Paragraph preceding the image/chart in document"
    )
    surrounding_text_after: str | None = Field(
        default=None, description="Paragraph following the image/chart in document"
    )
    extracted_data_points: dict[str, Any] = Field(
        default_factory=dict, description="Structured numerical data extracted from charts by VLM"
    )
    visual_entity_links: list[VisualEntityLink] = Field(default_factory=list)
    multi_vector_tokens: list[list[float]] | None = Field(
        default=None, description="Multi-vector representations for ColPali late-interaction"
    )


class PageVisualChunk(BaseChunk):
    """Page-level visual chunk representing an entire document page for ColPali late interaction."""

    modality: ModalityType = ModalityType.IMAGE
    page_number: int
    page_image_url: str
    patch_token_embeddings: list[list[float]] | None = None
    layout_elements: list[dict[str, Any]] = Field(default_factory=list)


class TableChunk(BaseChunk):
    """Structured table extracted from document."""

    modality: ModalityType = ModalityType.TABLE
    markdown_table: str
    csv_table: str | None = None
    summary: str | None = None
    section_title: str | None = None
    page_number: int | None = None


class Document(BaseModel):
    """Container for an ingested document and its parsed multimodal chunks."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_uri: str
    modality: ModalityType
    mime_type: str
    chunks: list[BaseChunk] = Field(default_factory=list)
    parent_chunks: list[ParentChunk] = Field(default_factory=list)
    media_assets: list[MediaAsset] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
