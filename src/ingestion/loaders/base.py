"""Abstract BaseLoader interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from src.domain.multimodal import Document


class BaseLoader(ABC):
    """Abstract base class for all file loaders."""

    @abstractmethod
    async def load(self, file_path: str | Path) -> Document:
        """Load and extract raw multimodal data from the given file."""
        pass
