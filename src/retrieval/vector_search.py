"""Vector search retriever querying dense text collections with Mixture of Domain Indices (MoDI) partitioning."""

from src.core.logging import logger
from src.domain.retrieval import DomainType, QueryRequest, SearchResult
from src.embeddings.text_embedder import TextEmbeddingEngine
from src.embeddings.vision_embedder import VisionEmbeddingEngine
from src.retrieval.base import BaseRetriever
from src.retrieval.domain_router import DomainClusterRouter
from src.storage.vector_store import BaseVectorStore


class VectorRetriever(BaseRetriever):
    """Retrieves similar chunks and images from vector store using dense embeddings and domain partitioning."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        text_embedder: TextEmbeddingEngine,
        vision_embedder: VisionEmbeddingEngine,
        domain_router: DomainClusterRouter | None = None,
    ):
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.vision_embedder = vision_embedder
        self.domain_router = domain_router or DomainClusterRouter(embedder=text_embedder)
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
            domain_buckets: dict[str, dict[str, list]] = {}

            for p in pages:
                cid = f"{p['document_id']}_p{p['page_number']}"
                ids.append(cid)
                texts.append(p["text"])
                payloads.append(p)

                # Classify page to its specialized domain
                doc_title = p.get("document_id", "")
                if "andrew ng" in doc_title.lower() or "deep learning" in doc_title.lower():
                    domain_key = DomainType.ENGINEERING_TECHNICAL.value
                else:
                    routing = self.domain_router.route_query(f"{doc_title}. {p['text'][:250]}")
                    domain_key = routing.primary_domain.value

                if domain_key not in domain_buckets:
                    domain_buckets[domain_key] = {"ids": [], "texts": [], "payloads": []}
                domain_buckets[domain_key]["ids"].append(cid)
                domain_buckets[domain_key]["texts"].append(p["text"])
                domain_buckets[domain_key]["payloads"].append(p)

            # Generate embeddings
            vectors = await self.text_embedder.get_batch_embeddings(texts)
            # 1. Insert into general base collection
            await self.vector_store.insert_vectors("multimodal_rag_text", ids, vectors, payloads)

            # 2. Insert into specialized MoDI domain collections
            id_to_vec = {cid: vec for cid, vec in zip(ids, vectors)}
            for dom_key, bdata in domain_buckets.items():
                dom_coll = f"modi_{dom_key}"
                dom_vecs = [id_to_vec[cid] for cid in bdata["ids"]]
                await self.vector_store.insert_vectors(dom_coll, bdata["ids"], dom_vecs, bdata["payloads"])
                logger.info(f"[MoDI] Partitioned {len(bdata['ids'])} vectors into specialized domain index: '{dom_coll}'")

            self._indexed = True
            logger.info(f"[VectorRetriever] Auto-indexed {len(ids)} dense vectors across domain partitions.")
        except Exception as e:
            logger.warning(f"[VectorRetriever] Failed to auto-index vectors: {e}")

    async def retrieve(self, request: QueryRequest) -> list[SearchResult]:
        logger.debug(f"Executing vector search for query: {request.query_text}")
        results: list[SearchResult] = []
        if not request.query_text:
            return results

        await self._ensure_vectors_indexed()
        vec = await self.text_embedder.get_embedding(request.query_text)

        # If a target domain is specified or can be inferred, target that domain collection first
        target_coll = None
        if request.target_domain:
            target_coll = f"modi_{request.target_domain}"

        if target_coll:
            try:
                results = await self.vector_store.search(target_coll, vec, top_k=request.top_k)
            except Exception:
                results = []

        if not results:
            # Fall back to base collection
            results = await self.vector_store.search("multimodal_rag_text", vec, top_k=request.top_k)

        return results
