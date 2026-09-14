"""Application configuration and environment settings."""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production application settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    APP_NAME: str = "Multimodal Graph RAG Engine"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "default-insecure-secret-key-change-in-production"

    # Task Queue & Cache (Redis)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Telemetry & Observability
    TELEMETRY_ENABLED: bool = False
    OTEL_SERVICE_NAME: str = "multimodal-graph-rag"
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # Neo4j Graph Database
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"
    NEO4J_DATABASE: str = "neo4j"
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = 50

    # Vector Database (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_TEXT: str = "multimodal_rag_text"
    QDRANT_COLLECTION_IMAGE: str = "multimodal_rag_images"

    # Media / Object Storage
    STORAGE_BACKEND: Literal["local", "s3", "minio"] = "local"
    LOCAL_STORAGE_PATH: str = "./data/media_store"
    S3_ENDPOINT_URL: str | None = None
    S3_ACCESS_KEY: str | None = None
    S3_SECRET_KEY: str | None = None
    S3_BUCKET_NAME: str = "multimodal-rag-assets"
    S3_REGION: str = "us-east-1"

    # LLM & Vision API Keys
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Groq & NVIDIA NIM Providers
    GROQ_API_KEY: str | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    NVIDIA_API_KEY: str | None = None
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

    # Default Models
    DEFAULT_LLM_MODEL: str = "qwen/qwen3.8-27b"
    DEFAULT_VLM_MODEL: str = "meta/llama-3.2-11b-vision-instruct"
    DEFAULT_TEXT_EMBEDDING_MODEL: str = "text-embedding-3-small"
    DEFAULT_VISION_EMBEDDING_MODEL: str = "clip-ViT-B/32"
    SPARSE_EMBEDDING_MODEL: str = "bm25"

    # Graph Extraction & Community Detection
    GRAPH_EXTRACTION_MAX_GLEANINGS: int = 1
    GRAPH_COMMUNITY_MAX_LEVEL: int = 3
    GRAPH_COMMUNITY_ALGORITHM: Literal["leiden", "louvain"] = "leiden"

    # Retrieval & Reranker Settings
    RETRIEVAL_TOP_K_VECTORS: int = 10
    RETRIEVAL_TOP_K_SPARSE: int = 10
    RETRIEVAL_TOP_K_GRAPH_NODES: int = 10
    RETRIEVAL_RRF_K: int = 60
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ENABLE_RERANKER: bool = True


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
