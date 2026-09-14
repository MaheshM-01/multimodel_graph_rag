"""Contextual Chunker (Anthropic Pattern prepending document and section context to prevent pronoun loss)."""

from src.core.logging import logger
from src.domain.multimodal import BaseChunk, ChildChunk, Document, TextChunk


class ContextualChunker:
    """Enriches isolated chunks with concise contextual headers (Anthropic Pattern).

    Eliminates pronoun ambiguity ('the company' -> 'Apple Inc.') and enhances Entity Resolution in GraphRAG.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def enrich_chunk(
        self,
        chunk: TextChunk | ChildChunk,
        document_title: str,
        section_title: str | None = None,
    ) -> str:
        """Generates and prepends a contextual prefix to the chunk content."""
        section_str = f", Section: {section_title}" if section_title else ""
        context_header = f"[Context: Document: {document_title}{section_str}]\n"

        enriched_content = f"{context_header}{chunk.content}"

        if isinstance(chunk, ChildChunk):
            chunk.contextual_header = context_header.strip()
            chunk.full_searchable_content = enriched_content

        return enriched_content

    async def enrich_all(self, document: Document, chunks: list[BaseChunk]) -> list[BaseChunk]:
        logger.debug(f"Applying Contextual Enrichment to {len(chunks)} chunks in '{document.title}'")
        for chunk in chunks:
            if isinstance(chunk, TextChunk):
                enriched = self.enrich_chunk(chunk, document.title, chunk.section_title)
                chunk.content = enriched
            elif isinstance(chunk, ChildChunk):
                self.enrich_chunk(chunk, document.title)

        return chunks
