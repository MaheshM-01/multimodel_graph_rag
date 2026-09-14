"""Multimodal Cross-Modal Chunker (Fuses visual charts and tables with surrounding textual context and VLM data)."""

from typing import Any
from src.core.logging import logger
from src.domain.multimodal import BoundingBox, Document, ImageChunk, MediaAsset, TableChunk


class CrossModalChunker:
    """Bridges images, charts, and tables with their surrounding textual context and VLM descriptions.

    Ensures that visual trends and numbers directly connect to entities in the Knowledge Graph.
    """

    def __init__(self):
        logger.info("Initialized CrossModalChunker for visual-textual bridging")

    async def create_cross_modal_image_chunk(
        self,
        document: Document,
        media_asset: MediaAsset,
        caption: str | None = None,
        surrounding_text_before: str | None = None,
        surrounding_text_after: str | None = None,
        bounding_box: BoundingBox | None = None,
        extracted_data_points: dict[str, Any] | None = None,
        vlm_summary: str | None = None,
    ) -> ImageChunk:
        """Constructs an ImageChunk enriched with surrounding textual paragraphs and structured VLM numbers."""
        logger.debug(f"Creating cross-modal chunk for visual asset '{media_asset.filename}'")

        chunk = ImageChunk(
            document_id=document.id,
            chunk_index=len(document.chunks),
            media_asset_id=media_asset.id,
            image_url=media_asset.file_path_or_url,
            caption=caption,
            vlm_summary=vlm_summary,
            bounding_box=bounding_box,
            surrounding_text_before=surrounding_text_before,
            surrounding_text_after=surrounding_text_after,
            extracted_data_points=extracted_data_points or {},
            metadata={
                "filename": media_asset.filename,
                "has_numerical_data": bool(extracted_data_points),
            },
        )
        return chunk

    async def create_cross_modal_table_chunk(
        self,
        document: Document,
        markdown_table: str,
        section_title: str | None = None,
        table_summary: str | None = None,
    ) -> TableChunk:
        """Constructs an atomic TableChunk with semantic summary and section context."""
        chunk = TableChunk(
            document_id=document.id,
            chunk_index=len(document.chunks),
            markdown_table=markdown_table,
            section_title=section_title,
            summary=table_summary,
        )
        return chunk
