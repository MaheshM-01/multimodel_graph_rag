"""Caching mechanisms for semantic queries and intermediate graph traversals."""

from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """Abstract cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        pass


class InMemoryCache(BaseCache):
    """Simple in-memory cache for fast local retrieval caching."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        return self._store.get(key)

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        self._store[key] = value
