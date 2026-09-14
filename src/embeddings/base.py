"""Abstract BaseEmbeddingEngine interface."""

from abc import ABC, abstractmethod


class BaseEmbeddingEngine(ABC):
    """Abstract interface for dense embedding generation."""

    @abstractmethod
    async def get_embedding(self, input_data: str | bytes) -> list[float]:
        """Generate embedding vector for text or media."""
        pass

    @abstractmethod
    async def get_batch_embeddings(self, inputs: list[str | bytes]) -> list[list[float]]:
        """Generate embeddings for a batch of inputs."""
        pass
