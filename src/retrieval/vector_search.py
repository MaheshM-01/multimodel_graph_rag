"""Vector search retriever querying dense text and image collections with auto-indexing."""

from src.core.logging import logger
from src.domain.retrieval import QueryRequest, SearchResult
from src.embeddings.text_embedder import TextEmbeddingEngine
from src.embeddings.vision_embedder import VisionEmbeddingEngine
from src.retrieval.base import BaseRetriever
from src.storage.vector_store import BaseVectorStore


class VectorRetriever(BaseRetriever):
    """Retrieves similar chunks and images from vector store using dense embeddings."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        text_embedder: TextEmbeddingEngine,
        vision_embedder: VisionEmbeddingEngine,
    ):
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.vision_embedder = vision_embedder
        self._indexed = False

    async def _ensure_vectors_indexed(self) -> None:
        if self._indexed:
            return

        try:
            from src.retrieval.document_search import get_document_search_engine
            engine = get_document_search_engine()
            pages = engine._ensure_documents_indexed()
            if not pages:
                return

            ids = []
            texts = []
            payloads = []
            for p in pages:
                cid = f"{p['document_id']}_p{p['page_number']}"
                ids.append(cid)
                texts.append(p["text"])
                payloads.append(p)

            # Generate embeddings
            vectors = await self.text_embedder.get_batch_embeddings(texts)
            await self.vector_store.insert_vectors("multimodal_rag_text", ids, vectors, payloads)
            self._indexed = True
            logger.info(f"[VectorRetriever] Auto-indexed {len(ids)} dense vectors into 'multimodal_rag_text'")
        except Exception as e:
            logger.warning(f"[VectorRetriever] Failed to auto-index vectors: {e}")

    async def retrieve(self, request: QueryRequest) -> list[SearchResult]:
        logger.debug(f"Executing vector search for query: {request.query_text}")
        results: list[SearchResult] = []
        if request.query_text:
            await self._ensure_vectors_indexed()
            vec = await self.text_embedder.get_embedding(request.query_text)
            results = await self.vector_store.search("multimodal_rag_text", vec, top_k=request.top_k)
        return results
