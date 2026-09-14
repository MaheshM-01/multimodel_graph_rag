"""Unit tests verifying the Top 6 Advanced Chunking Strategies."""

import pytest
from src.core.constants import ModalityType
from src.domain.multimodal import (
    BoundingBox,
    ChildChunk,
    Document,
    MediaAsset,
    TextChunk,
)
from src.ingestion.chunkers.contextual_chunker import ContextualChunker
from src.ingestion.chunkers.cross_modal_chunker import CrossModalChunker
from src.ingestion.chunkers.hierarchical_chunker import HierarchicalChunker
from src.ingestion.chunkers.layout_aware_chunker import LayoutAwareChunker
from src.ingestion.chunkers.multimodal_chunker import MultimodalChunker
from src.ingestion.chunkers.page_visual_chunker import PageVisualChunker
from src.ingestion.chunkers.propositional_chunker import PropositionalChunker


@pytest.mark.asyncio
async def test_layout_aware_chunker_preserves_tables():
    raw_md = """# Financial Highlights
The company delivered strong revenue growth across all units.

| Quarter | Revenue (B) | Growth |
|---------|-------------|--------|
| Q1      | $25.4       | +12%   |
| Q2      | $28.1       | +15%   |

Operating income also reached new records.
"""
    doc = Document(
        title="10-Q Report",
        source_uri="test://report.md",
        modality=ModalityType.TEXT,
        mime_type="text/markdown",
        metadata={"raw_text": raw_md},
    )

    chunker = LayoutAwareChunker()
    chunks = await chunker.chunk(doc)

    assert len(chunks) == 3
    # Check that table is its own atomic chunk
    assert chunks[1].modality == ModalityType.TABLE
    assert "| Quarter | Revenue (B) | Growth |" in chunks[1].markdown_table


@pytest.mark.asyncio
async def test_contextual_chunker_prepends_headers():
    doc = Document(
        title="Apple Q3 Report",
        source_uri="test://apple.pdf",
        modality=ModalityType.TEXT,
        mime_type="application/pdf",
    )
    child = ChildChunk(
        document_id=doc.id,
        chunk_index=0,
        parent_chunk_id="parent_1",
        content="The company grew by 20% in Greater China.",
    )

    enricher = ContextualChunker()
    enriched = enricher.enrich_chunk(child, document_title=doc.title, section_title="iPhone Sales")

    assert "[Context: Document: Apple Q3 Report, Section: iPhone Sales]" in enriched
    assert child.contextual_header is not None


@pytest.mark.asyncio
async def test_hierarchical_parent_child_linking():
    doc = Document(
        title="Cloud Architecture",
        source_uri="test://arch.pdf",
        modality=ModalityType.TEXT,
        mime_type="application/pdf",
    )
    long_text = " ".join(["Database optimization is critical for performance."] * 100)
    tc = TextChunk(
        document_id=doc.id,
        chunk_index=0,
        content=long_text,
        section_title="Storage Layer",
    )

    hierarchical = HierarchicalChunker(parent_size=500, child_size=50, child_overlap=10)
    parents, children = await hierarchical.split_hierarchical(doc, [tc])

    assert len(parents) == 1
    assert len(children) > 1
    # Check relational foreign key link
    for child in children:
        assert child.parent_chunk_id == parents[0].id
        assert child.id in parents[0].child_chunk_ids


@pytest.mark.asyncio
async def test_cross_modal_chunker_enrichment():
    doc = Document(
        title="AI Hardware Overview",
        source_uri="test://doc.pdf",
        modality=ModalityType.TEXT,
        mime_type="application/pdf",
    )
    asset = MediaAsset(
        filename="gpu_chart.png",
        modality=ModalityType.IMAGE,
        mime_type="image/png",
        file_path_or_url="s3://assets/gpu_chart.png",
    )
    bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.8, page_number=3)

    cross_chunker = CrossModalChunker()
    img_chunk = await cross_chunker.create_cross_modal_image_chunk(
        document=doc,
        media_asset=asset,
        caption="GPU Throughput vs Batch Size",
        surrounding_text_before="As demonstrated in the experimental benchmarks below:",
        bounding_box=bbox,
        extracted_data_points={"peak_tflops": 1979, "batch_size": 128},
    )

    assert img_chunk.modality == ModalityType.IMAGE
    assert img_chunk.bounding_box.page_number == 3
    assert img_chunk.extracted_data_points["peak_tflops"] == 1979
    assert "experimental benchmarks" in img_chunk.surrounding_text_before


@pytest.mark.asyncio
async def test_propositional_chunker_atomic_facts():
    doc = Document(
        title="Company History",
        source_uri="test://history.txt",
        modality=ModalityType.TEXT,
        mime_type="text/plain",
    )
    tc = TextChunk(
        document_id=doc.id,
        chunk_index=0,
        content="Sundar Pichai, who joined Google in 2004, became CEO in 2015 after leading Android and Chrome.",
    )

    prop_chunker = PropositionalChunker()
    propositions = await prop_chunker.extract_propositions(doc, tc)

    assert len(propositions) >= 2
    for prop in propositions:
        assert prop.proposition_statement.endswith(".")
        assert prop.source_chunk_id == tc.id


@pytest.mark.asyncio
async def test_composite_multimodal_chunker():
    raw_md = """# Executive Summary
Cloud Services revenue expanded rapidly due to AI adoption.

| Year | Revenue |
|------|---------|
| 2024 | $30B    |
| 2025 | $45B    |

Further investments are planned for infrastructure.
"""
    doc = Document(
        title="Annual Strategy",
        source_uri="test://strategy.md",
        modality=ModalityType.TEXT,
        mime_type="text/markdown",
        metadata={"raw_text": raw_md},
        media_assets=[
            MediaAsset(
                filename="growth_curve.png",
                modality=ModalityType.IMAGE,
                mime_type="image/png",
                file_path_or_url="s3://bucket/growth.png",
            )
        ],
    )

    composite = MultimodalChunker()
    chunks = await composite.chunk(doc)

    assert len(chunks) > 0
    assert len(doc.parent_chunks) > 0
    # Confirm both child text chunks and visual image chunks are present
    modalities = {c.modality for c in chunks}
    assert ModalityType.TEXT in modalities
    assert ModalityType.IMAGE in modalities
