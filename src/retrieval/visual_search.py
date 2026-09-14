"""Multimodal / Visual Search Retriever with auto-indexed visual pages and diagram groundings."""

from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.retrieval import QueryRequest, SearchResult
from src.embeddings.vision_embedder import VisionEmbeddingEngine
from src.retrieval.base import BaseRetriever
from src.storage.vector_store import BaseVectorStore


class VisualRetriever(BaseRetriever):
    """Retrieves visual artifacts (charts, diagrams, infographics, page images) matching text or image queries."""

    def __init__(self, vector_store: BaseVectorStore, vision_embedder: VisionEmbeddingEngine):
        self.vector_store = vector_store
        self.vision_embedder = vision_embedder
        self._indexed = False
        logger.info("Initialized VisualRetriever for multimodal visual searching")

    async def _ensure_visuals_indexed(self) -> None:
        if self._indexed:
            return

        try:
            from src.retrieval.document_search import get_document_search_engine
            engine = get_document_search_engine()
            pages = engine._ensure_documents_indexed()
            visual_pages = [p for p in pages if p.get("has_images")]

            if not visual_pages:
                return

            ids = []
            captions = []
            payloads = []
            for p in visual_pages:
                cid = f"visual_{p['document_id']}_p{p['page_number']}"
                ids.append(cid)
                caption = f"Diagram and visual evidence from page {p['page_number']}: {p['text'][:200]}"
                captions.append(caption)
                payloads.append({
                    **p,
                    "modality": "image",
                    "caption": caption,
                })

            vectors = await self.vision_embedder.get_batch_embeddings(captions)
            await self.vector_store.insert_vectors("multimodal_rag_images", ids, vectors, payloads)
            self._indexed = True
            logger.info(f"[VisualRetriever] Auto-indexed {len(ids)} visual pages into 'multimodal_rag_images'")
        except Exception as e:
            logger.warning(f"[VisualRetriever] Failed to auto-index visuals: {e}")

    async def retrieve(self, request: QueryRequest) -> list[SearchResult]:
        logger.debug(f"Executing visual / ColPali search for query: '{request.query_text}'")
        results: list[SearchResult] = []

        query_input = request.query_image_url_or_path or request.query_text
        if not query_input:
            return results

        await self._ensure_visuals_indexed()

        # 1. Generate vision/cross-modal embedding vector
        vision_vec = await self.vision_embedder.get_embedding(query_input)

        # 2. Query Qdrant visual collection
        raw_results = await self.vector_store.search(
            collection="multimodal_rag_images",
            query_vector=vision_vec,
            top_k=request.top_k,
        )

        for r in raw_results:
            results.append(
                SearchResult(
                    id=r.id,
                    modality=ModalityType.IMAGE,
                    score=r.score,
                    content=r.content or f"Visual diagram / chart evidence from {r.metadata.get('document_id', 'doc')}",
                    image_url=r.image_url,
                    source_type="visual",
                    data_points=r.data_points,
                    metadata=r.metadata,
                )
            )

        return results
