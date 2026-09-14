"""Domain-specific exceptions for Multimodal Graph RAG."""


class MultimodalRAGException(Exception):
    """Base exception class for all Multimodal Graph RAG errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(MultimodalRAGException):
    """Raised when environment or settings configuration is invalid."""


class GraphDatabaseError(MultimodalRAGException):
    """Raised when a graph database operation fails."""


class VectorDatabaseError(MultimodalRAGException):
    """Raised when vector storage or search operations fail."""


class StorageError(MultimodalRAGException):
    """Raised when media or blob storage fails."""


class IngestionError(MultimodalRAGException):
    """Raised during document ingestion, parsing, or chunking."""


class EmbeddingError(MultimodalRAGException):
    """Raised when multimodal embedding generation fails."""


class RetrievalError(MultimodalRAGException):
    """Raised during hybrid search or graph traversal."""


class GenerationError(MultimodalRAGException):
    """Raised during LLM/VLM generation or citation assembly."""
