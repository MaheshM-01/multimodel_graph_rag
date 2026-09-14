"""Abstract BaseChunker interface."""

from abc import ABC, abstractmethod
from src.domain.multimodal import BaseChunk, Document


class BaseChunker(ABC):
    """Abstract base class for chunking multimodal documents."""

    @abstractmethod
    async def chunk(self, document: Document) -> list[BaseChunk]:
        """Split a document into multimodal chunks (text, image, table)."""
        pass
