"""Abstract BaseRetriever interface."""

from abc import ABC, abstractmethod
from src.domain.retrieval import QueryRequest, SearchResult


class BaseRetriever(ABC):
    """Abstract interface for all retrieval strategies."""

    @abstractmethod
    async def retrieve(self, request: QueryRequest) -> list[SearchResult]:
        """Execute retrieval and return scored results."""
        pass
