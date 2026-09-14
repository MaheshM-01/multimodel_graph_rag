"""Demonstration script for multimodal document ingestion and graph construction."""

import asyncio
from src.core.constants import ModalityType
from src.core.logging import logger, setup_logging
from src.domain.graph import EntityNode, GraphEdge
from src.domain.multimodal import Document, MediaAsset, TextChunk
from src.graph.client import get_graph_client
from src.graph.construction.builder import GraphBuilder


async def run_demo():
    setup_logging("INFO")
    logger.info("Starting Multimodal Graph RAG demonstration ingestion...")

    # 1. Mock multimodal document
    doc = Document(
        title="Quarterly Revenue & Strategy Report",
        source_uri="local://reports/q3_strategy.pdf",
        modality=ModalityType.TEXT,
        mime_type="application/pdf",
        media_assets=[
            MediaAsset(
                filename="revenue_chart.png",
                modality=ModalityType.IMAGE,
                mime_type="image/png",
                file_path_or_url="local://data/media_store/revenue_chart.png",
            )
        ],
    )

    # 2. Extracted sample entities & cross-modal edges
    entities = [
        EntityNode(name="Cloud Services", entity_type="Product", description="Next-gen cloud computing division"),
        EntityNode(name="Revenue Growth", entity_type="FinancialMetric", description="35% Year-over-Year revenue expansion"),
    ]
    edges = [
        GraphEdge(
            source_id="Cloud Services",
            target_id="Revenue Growth",
            relationship_type="DRIVES",
            description="Cloud services division drove significant revenue growth",
        )
    ]

    client = get_graph_client()
    try:
        await client.connect()
        builder = GraphBuilder(client)
        await builder.ingest_document_graph(doc, entities, edges)
        logger.info("Demo document and graph entities ingested successfully!")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
