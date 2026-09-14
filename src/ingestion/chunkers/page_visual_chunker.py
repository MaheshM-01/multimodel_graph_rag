"""Page-Level Visual Chunker (ColPali / Late-Interaction approach for complex multi-column documents & forms)."""

from src.core.logging import logger
from src.domain.multimodal import Document, PageVisualChunk


class PageVisualChunker:
    """Chunks documents at the page image level for ColPali / SigLIP late-interaction multi-vector indexing.

    Preserves exact spatial layouts, multi-column flows, forms, and charts without OCR loss.
    """

    def __init__(self, patch_dim: int = 128):
        self.patch_dim = patch_dim

    async def chunk_pages(self, document: Document) -> list[PageVisualChunk]:
        """Creates PageVisualChunk instances for each page in the document."""
        logger.info(f"Extracting Page-Level Visual chunks for document '{document.title}'")
        page_chunks: list[PageVisualChunk] = []

        # If document contains page media assets or screenshots
        for idx, asset in enumerate(document.media_assets):
            page_chunk = PageVisualChunk(
                document_id=document.id,
                chunk_index=idx,
                page_number=idx + 1,
                page_image_url=asset.file_path_or_url,
                metadata={"filename": asset.filename, "size_bytes": asset.size_bytes},
            )
            page_chunks.append(page_chunk)

        return page_chunks
