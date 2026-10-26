"""Mixture of Domain Indices (MoDI) Manager.

Maintains partitioned vector & document indices per domain to eliminate cross-domain noise,
semantic drift, and false positives across multi-domain enterprise knowledge bases.
"""

from typing import Any, Dict, List, Optional
from src.core.logging import logger
from src.domain.retrieval import DomainType, SearchResult
from src.retrieval.domain_router import DomainClusterRouter
from src.storage.vector_store import BaseVectorStore


class MixtureOfDomainIndicesManager:
    """Orchestrates ingestion, partitioning, and retrieval across specialized domain indices."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        domain_router: Optional[DomainClusterRouter] = None,
    ):
        self.vector_store = vector_store
        self.domain_router = domain_router or DomainClusterRouter()
        self._domain_counts: Dict[str, int] = {dom.value: 0 for dom in DomainType}
        logger.info("[MoDI] Initialized MixtureOfDomainIndicesManager with 6 domain namespaces.")

    def get_collection_name(self, domain: DomainType) -> str:
        """Return collection identifier for a specific domain."""
        return f"modi_{domain.value}"

    def classify_document(self, title: str, sample_text: str) -> DomainType:
        """Classify a document or chunk to its target domain index using embedding similarity."""
        combined_text = f"{title}. {sample_text[:1200]}"
        routing = self.domain_router.route_query(combined_text)
        return routing.primary_domain

    async def insert_domain_vectors(
        self,
        domain: DomainType,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Insert vectors directly into a specialized domain index."""
        collection_name = self.get_collection_name(domain)
        await self.vector_store.insert_vectors(collection_name, ids, vectors, payloads)
        self._domain_counts[domain.value] = self._domain_counts.get(domain.value, 0) + len(ids)
        logger.info(f"[MoDI] Indexed {len(ids)} items into domain collection '{collection_name}'")

    async def search_domain_indices(
        self,
        active_collections: list[str],
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search strictly across the active domain collections pruned by DomainClusterRouter."""
        all_results: list[SearchResult] = []
        seen_ids = set()

        for coll in active_collections:
            try:
                results = await self.vector_store.search(coll, query_vector, top_k=top_k)
                for res in results:
                    if res.id not in seen_ids:
                        seen_ids.add(res.id)
                        res.metadata["domain_collection"] = coll
                        all_results.append(res)
            except Exception as e:
                logger.warning(f"[MoDI] Search failed on collection '{coll}': {e}")

        # Sort combined results by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def get_domain_stats(self) -> dict[str, Any]:
        """Return distribution of indexed items across specialized domain indices."""
        return {
            "total_domains": len(DomainType) - 1,  # Exclude CROSS_DOMAIN_ALL
            "domain_namespaces": [f"modi_{d.value}" for d in DomainType if d != DomainType.CROSS_DOMAIN_ALL],
            "counts": self._domain_counts,
        }
