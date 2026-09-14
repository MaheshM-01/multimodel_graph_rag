"""BaseParser abstract class."""

from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """Abstract base class for document parsing components."""

    @abstractmethod
    async def parse(self, input_data: Any) -> Any:
        """Parse the input data into structured representation."""
        pass
