"""Unit tests for domain models and data integrity."""

import pytest
from src.core.constants import ModalityType, NodeType
from src.domain.graph import EntityNode, GraphEdge
from src.domain.multimodal import BoundingBox, Document, ImageChunk, MediaAsset, TextChunk


def test_bounding_box_coordinates():
    bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9, page_number=2)
    assert bbox.x_min == 0.1
    assert bbox.y_max == 0.9
    assert bbox.page_number == 2


def test_document_and_chunks():
    doc = Document(
        title="Architecture Overview",
        source_uri="/docs/arch.pdf",
        modality=ModalityType.TEXT,
        mime_type="application/pdf",
    )
    assert doc.id is not None
    assert doc.title == "Architecture Overview"

    text_chunk = TextChunk(
        document_id=doc.id,
        chunk_index=0,
        content="Graph RAG enhances reasoning.",
    )
    assert text_chunk.modality == ModalityType.TEXT
    assert text_chunk.content == "Graph RAG enhances reasoning."


def test_entity_node_and_edge():
    node = EntityNode(
        name="Neo4j",
        entity_type="Database",
        description="Graph Database Management System",
    )
    assert node.label == NodeType.ENTITY
    assert node.name == "Neo4j"

    edge = GraphEdge(
        source_id="App",
        target_id="Neo4j",
        relationship_type="CONNECTS_TO",
    )
    assert edge.relationship_type == "CONNECTS_TO"
