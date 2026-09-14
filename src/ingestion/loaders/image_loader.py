"""Image loader for visual artifacts, photos, diagrams, and charts."""

from pathlib import Path
from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.multimodal import Document, MediaAsset
from src.ingestion.loaders.base import BaseLoader


class ImageLoader(BaseLoader):
    """Loads standalone image files (PNG, JPG, WEBP, etc.)."""

    async def load(self, file_path: str | Path) -> Document:
        path = Path(file_path)
        logger.info(f"Loading image document: {path.name}")

        media_asset = MediaAsset(
            filename=path.name,
            modality=ModalityType.IMAGE,
            mime_type=f"image/{path.suffix.lstrip('.') or 'jpeg'}",
            file_path_or_url=str(path.resolve()),
            size_bytes=path.stat().st_size if path.exists() else 0,
        )

        doc = Document(
            title=path.stem,
            source_uri=str(path.resolve()),
            modality=ModalityType.IMAGE,
            mime_type=media_asset.mime_type,
            media_assets=[media_asset],
            metadata={"filename": path.name},
        )
        return doc
