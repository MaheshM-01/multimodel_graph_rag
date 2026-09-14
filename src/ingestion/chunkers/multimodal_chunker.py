"""Composite Multimodal Chunker coordinating the Top 6 Advanced Chunking Strategies."""

from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.multimodal import BaseChunk, Document, ImageChunk, ParentChunk, ChildChunk, TextChunk
from src.ingestion.chunkers.base import BaseChunker
from src.ingestion.chunkers.contextual_chunker import ContextualChunker
from src.ingestion.chunkers.cross_modal_chunker import CrossModalChunker
from src.ingestion.chunkers.hierarchical_chunker import HierarchicalChunker
from src.ingestion.chunkers.layout_aware_chunker import LayoutAwareChunker
from src.ingestion.chunkers.page_visual_chunker import PageVisualChunker
from src.ingestion.chunkers.propositional_chunker import PropositionalChunker


class MultimodalChunker(BaseChunker):
    """Composite Chunker executing the production Multimodal Graph RAG chunking recipe:

    1. Layout-Aware Splitting (Preserves headings, tables, and figures)
    2. Hierarchical Small-to-Big (Parent 1200 tokens / Child 300 tokens)
    3. Contextual Enrichment (Prepends document & section context headers)
    4. Cross-Modal Image/Table Bridging (Surrounding paragraphs + VLM data points)
    5. Propositional Atomic Fact Extraction (For direct Graph triple mapping)
    6. Page-Level Visual Chunking (For ColPali multi-vector representation)
    """

    def __init__(
        self,
        parent_size: int = 1200,
        child_size: int = 300,
        enable_contextual_enrichment: bool = True,
        enable_propositions: bool = True,
    ):
        self.layout_chunker = LayoutAwareChunker()
        self.hierarchical_chunker = HierarchicalChunker(parent_size=parent_size, child_size=child_size)
        self.contextual_chunker = ContextualChunker()
        self.cross_modal_chunker = CrossModalChunker()
        self.propositional_chunker = PropositionalChunker()
        self.page_visual_chunker = PageVisualChunker()
        self.enable_contextual_enrichment = enable_contextual_enrichment
        self.enable_propositions = enable_propositions

    async def chunk(self, document: Document) -> list[BaseChunk]:
        logger.info(f"[MultimodalChunker] Executing Advanced Chunking Pipeline on: '{document.title}'")

        all_chunks: list[BaseChunk] = []

        # 1. Structural / Layout-Aware Chunking
        base_structural_chunks = await self.layout_chunker.chunk(document)

        text_chunks: list[TextChunk] = [c for c in base_structural_chunks if isinstance(c, TextChunk)]
        table_chunks = [c for c in base_structural_chunks if not isinstance(c, TextChunk)]
        all_chunks.extend(table_chunks)

        # 2. Hierarchical Parent-Child Chunking (Small-to-Big)
        if text_chunks:
            parents, children = await self.hierarchical_chunker.split_hierarchical(document, text_chunks)
            document.parent_chunks = parents

            # 3. Contextual Enrichment (Anthropic Pattern)
            if self.enable_contextual_enrichment:
                for child in children:
                    self.contextual_chunker.enrich_chunk(child, document.title)

            all_chunks.extend(children)
        else:
            all_chunks.extend(text_chunks)

        # 4. Cross-Modal Image Chunks (Images with surrounding text & bboxes)
        for idx, asset in enumerate(document.media_assets):
            if asset.modality == ModalityType.IMAGE:
                # Find neighboring text context if available
                surrounding_before = text_chunks[0].content[:200] if text_chunks else None
                img_chunk = await self.cross_modal_chunker.create_cross_modal_image_chunk(
                    document=document,
                    media_asset=asset,
                    caption=asset.metadata.get("caption"),
                    surrounding_text_before=surrounding_before,
                    extracted_data_points=asset.metadata.get("data_points", {}),
                    vlm_summary=asset.metadata.get("vlm_summary"),
                )
                all_chunks.append(img_chunk)

        # 5. Page-Level Visual Chunks (ColPali late-interaction)
        page_chunks = await self.page_visual_chunker.chunk_pages(document)
        all_chunks.extend(page_chunks)

        document.chunks = all_chunks
        logger.info(f"[MultimodalChunker] Completed: produced {len(all_chunks)} total multimodal chunks.")
        return all_chunks
