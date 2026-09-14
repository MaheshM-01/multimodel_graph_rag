"""System constants and enumerations for multimodal graph entities."""

from enum import Enum


class ModalityType(str, Enum):
    """Supported modalities in the Multimodal Graph RAG system."""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    AUDIO = "audio"
    VIDEO = "video"


class NodeType(str, Enum):
    """Knowledge graph node labels."""

    DOCUMENT = "Document"
    SECTION = "Section"
    PARENT_CHUNK = "ParentChunk"
    CHILD_CHUNK = "ChildChunk"
    CHUNK = "Chunk"
    IMAGE = "Image"
    TABLE = "Table"
    ENTITY = "Entity"
    COMMUNITY = "Community"
    PROPOSITION = "Proposition"
    PAGE_VISUAL = "PageVisual"


class EdgeType(str, Enum):
    """Knowledge graph relationship types."""

    HAS_SECTION = "HAS_SECTION"
    CONTAINS_CHUNK = "CONTAINS_CHUNK"
    BELONGS_TO_PARENT = "BELONGS_TO_PARENT"
    DEPICTS = "DEPICTS"
    MENTIONS = "MENTIONS"
    RELATES_TO = "RELATES_TO"
    BELONGS_TO_COMMUNITY = "BELONGS_TO_COMMUNITY"
    CROSS_MODAL_LINK = "CROSS_MODAL_LINK"
    NEXT_CHUNK = "NEXT_CHUNK"
    VISUALLY_REPRESENTS = "VISUALLY_REPRESENTS"
    LOCATED_IN_PAGE = "LOCATED_IN_PAGE"
    CHARTED_IN = "CHARTED_IN"
    EXTRACTED_FROM_PROPOSITION = "EXTRACTED_FROM_PROPOSITION"


# Default vector dimensions
DIMENSION_CLIP_VIT_B32 = 512
DIMENSION_TEXT_EMBEDDING_3_SMALL = 1536
DIMENSION_TEXT_EMBEDDING_3_LARGE = 3072
