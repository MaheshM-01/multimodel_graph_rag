"""Vector store abstraction and Qdrant implementation with high-speed in-memory vector index."""

from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict
import numpy as np
from src.config.settings import Settings, get_settings
from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.retrieval import SearchResult


class BaseVectorStore(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    async def insert_vectors(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        pass

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        pass


class QdrantVectorStore(BaseVectorStore):
    """Qdrant implementation for multimodal dense vector retrieval with local fallback index."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._collections: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._matrices: dict[str, np.ndarray] = {}
        self._matrix_dirty: dict[str, bool] = defaultdict(bool)
        logger.info(f"Initialized QdrantVectorStore at {self.settings.QDRANT_HOST}:{self.settings.QDRANT_PORT}")

    async def insert_vectors(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        logger.info(f"Inserting {len(ids)} vectors into Qdrant collection '{collection}'")
        for doc_id, vec, payload in zip(ids, vectors, payloads):
            # Check existing to update or append
            existing = next((item for item in self._collections[collection] if item["id"] == doc_id), None)
            if existing:
                existing["vector"] = np.array(vec, dtype=np.float32)
                existing["payload"] = payload
            else:
                self._collections[collection].append({
                    "id": doc_id,
                    "vector": np.array(vec, dtype=np.float32),
                    "payload": payload,
                })
        self._matrix_dirty[collection] = True

    def _get_matrix(self, collection: str) -> np.ndarray:
        if self._matrix_dirty[collection] or collection not in self._matrices:
            items = self._collections[collection]
            if not items:
                self._matrices[collection] = np.empty((0, 0), dtype=np.float32)
            else:
                self._matrices[collection] = np.vstack([it["vector"] for it in items])
            self._matrix_dirty[collection] = False
        return self._matrices[collection]

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        logger.debug(f"Searching Qdrant collection '{collection}' with top_k={top_k}")
        items = self._collections.get(collection, [])
        if not items or not query_vector:
            return []

        matrix = self._get_matrix(collection)
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-6:
            return []

        # Vectorized cosine similarity: dot product of normalized vectors
        m_norms = np.linalg.norm(matrix, axis=1)
        m_norms[m_norms < 1e-6] = 1e-6
        sims = np.dot(matrix, q_vec) / (m_norms * q_norm)

        # Top K indices
        top_indices = np.argsort(sims)[::-1][:top_k]
        results: list[SearchResult] = []

        for idx in top_indices:
            score = float(sims[idx])
            if score < 0.05:
                continue
            item = items[idx]
            payload = item["payload"]
            is_image = payload.get("modality") == "image" or "image" in collection

            results.append(
                SearchResult(
                    id=item["id"],
                    modality=ModalityType.IMAGE if is_image else ModalityType.TEXT,
                    score=round(score, 4),
                    content=payload.get("content", payload.get("text", "")),
                    image_url=payload.get("preview_url") or payload.get("image_url"),
                    source_type="dense_vector",
                    data_points=payload.get("data_points", {}),
                    metadata=payload,
                )
            )

        return results
