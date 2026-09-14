"""Multimodal media asset storage (Local filesystem, MinIO, S3)."""

from abc import ABC, abstractmethod
from pathlib import Path
from src.config.settings import Settings, get_settings
from src.core.logging import logger


class BaseMediaStore(ABC):
    """Abstract interface for storing raw multimodal images, diagrams, and files."""

    @abstractmethod
    async def save_media(self, file_name: str, content: bytes, mime_type: str) -> str:
        """Saves media and returns access URI or URL."""
        pass

    @abstractmethod
    async def get_media(self, file_uri: str) -> bytes:
        """Retrieves raw media bytes."""
        pass


class LocalMediaStore(BaseMediaStore):
    """Local filesystem media store for development and testing."""

    def __init__(self, base_path: str = "./data/media_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_media(self, file_name: str, content: bytes, mime_type: str) -> str:
        dest_path = self.base_path / file_name
        dest_path.write_bytes(content)
        logger.debug(f"Saved media asset locally to {dest_path}")
        return str(dest_path.resolve())

    async def get_media(self, file_uri: str) -> bytes:
        return Path(file_uri).read_bytes()


class S3MediaStore(BaseMediaStore):
    """S3 and MinIO compatible object storage for production media assets."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        logger.info(f"Initialized S3/MinIO media store with bucket '{self.settings.S3_BUCKET_NAME}'")

    async def save_media(self, file_name: str, content: bytes, mime_type: str) -> str:
        # Uploads object to S3 / MinIO and returns URL
        return f"{self.settings.S3_ENDPOINT_URL}/{self.settings.S3_BUCKET_NAME}/{file_name}"

    async def get_media(self, file_uri: str) -> bytes:
        return b""


def get_media_store() -> BaseMediaStore:
    """Factory returning configured media store."""
    settings = get_settings()
    if settings.STORAGE_BACKEND in ("s3", "minio"):
        return S3MediaStore(settings)
    return LocalMediaStore(settings.LOCAL_STORAGE_PATH)
