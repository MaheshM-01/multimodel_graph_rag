"""Table parser for converting tables to Markdown/CSV and semantic summaries."""

from typing import Any
from src.core.logging import logger
from src.domain.multimodal import TableChunk
from src.ingestion.parsers.base import BaseParser


class TableParser(BaseParser):
    """Parses tabular structures from documents into TableChunk instances."""

    async def parse(self, table_raw_data: Any) -> TableChunk | None:
        logger.debug("Parsing document table")
        return None
