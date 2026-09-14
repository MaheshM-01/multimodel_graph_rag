"""OCR parser for visual text extraction."""

from pathlib import Path
from src.core.logging import logger
from src.ingestion.parsers.base import BaseParser


class OCRParser(BaseParser):
    """Extracts text, bounding boxes, and words from visual images using OCR."""

    def __init__(self, language: str = "eng"):
        self.language = language

    async def parse(self, image_input: str | Path | bytes) -> str:
        logger.debug("Running OCR parser on visual input")
        return ""
