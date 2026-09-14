"""Layout-Aware / Structural Chunker (DOM-based splitting preserving sections, tables, and figures)."""

import re
from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.multimodal import BaseChunk, Document, TableChunk, TextChunk
from src.ingestion.chunkers.base import BaseChunker


class LayoutAwareChunker(BaseChunker):
    """Parses structural document elements (Headings, Paragraphs, Tables, Figures).

    Never splits tables or figures across chunk boundaries.
    """

    def __init__(self, max_tokens: int = 1000):
        self.max_tokens = max_tokens

    async def chunk(self, document: Document) -> list[BaseChunk]:
        logger.info(f"Executing Layout-Aware structural chunking on: '{document.title}'")
        chunks: list[BaseChunk] = []

        # Process any existing structured chunks (tables, images)
        for chunk in document.chunks:
            if isinstance(chunk, TableChunk):
                chunks.append(chunk)

        # Extract textual sections based on markdown/heading DOM structure
        raw_text = document.metadata.get("raw_text", "")
        if not raw_text and document.chunks:
            return document.chunks

        sections = self._split_by_headings(raw_text)

        for idx, (section_title, section_body) in enumerate(sections):
            # Check if section body contains a table block
            table_match = re.search(r"(\|.*\|\n\|[-:\s|]+\|\n(?:\|.*\|\n?)+)", section_body)
            if table_match:
                table_md = table_match.group(1)
                text_before = section_body[: table_match.start()].strip()
                text_after = section_body[table_match.end() :].strip()

                if text_before:
                    chunks.append(
                        TextChunk(
                            document_id=document.id,
                            chunk_index=len(chunks),
                            content=text_before,
                            section_title=section_title,
                        )
                    )

                chunks.append(
                    TableChunk(
                        document_id=document.id,
                        chunk_index=len(chunks),
                        markdown_table=table_md,
                        section_title=section_title,
                    )
                )

                if text_after:
                    chunks.append(
                        TextChunk(
                            document_id=document.id,
                            chunk_index=len(chunks),
                            content=text_after,
                            section_title=section_title,
                        )
                    )
            else:
                chunks.append(
                    TextChunk(
                        document_id=document.id,
                        chunk_index=len(chunks),
                        content=section_body,
                        section_title=section_title,
                    )
                )

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str, str]]:
        """Split text by Markdown headings (#, ##, ###) while preserving section titles."""
        heading_pattern = re.compile(r"^(#{1,4}\s+.+)$", re.MULTILINE)
        splits = heading_pattern.split(text)

        sections: list[tuple[str, str]] = []
        current_title = "Introduction"

        if splits and not splits[0].startswith("#"):
            initial_text = splits.pop(0).strip()
            if initial_text:
                sections.append((current_title, initial_text))

        for i in range(0, len(splits), 2):
            if i + 1 < len(splits):
                title = splits[i].lstrip("#").strip()
                content = splits[i + 1].strip()
                if content:
                    sections.append((title, content))

        return sections or [("General", text)]
