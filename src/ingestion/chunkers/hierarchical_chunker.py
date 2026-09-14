"""Hierarchical / Parent-Child Chunker (Small-to-Big Retrieval for Multimodal Graph RAG)."""

from src.core.logging import logger
from src.domain.multimodal import ChildChunk, Document, ParentChunk, TextChunk


class HierarchicalChunker:
    """Generates Hierarchical Parent-Child chunk pairs.

    - Child Chunks (200-300 tokens): Indexed in Vector DB (Qdrant) for high similarity precision.
    - Parent Chunks (1000-1500 tokens): Stored in Knowledge Graph (Neo4j) for rich reasoning context.
    """

    def __init__(self, parent_size: int = 1200, child_size: int = 300, child_overlap: int = 40):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    async def split_hierarchical(
        self, document: Document, text_chunks: list[TextChunk]
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        logger.info(f"Generating Hierarchical Parent-Child chunks for document '{document.title}'")

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        # Group text chunks by section
        sections: dict[str, list[TextChunk]] = {}
        for tc in text_chunks:
            sec = tc.section_title or "General"
            sections.setdefault(sec, []).append(tc)

        for sec_title, sec_chunks in sections.items():
            full_sec_text = "\n\n".join(c.content for c in sec_chunks)
            words = full_sec_text.split()

            # Create Parent Chunk
            parent = ParentChunk(
                document_id=document.id,
                chunk_index=len(parents),
                section_title=sec_title,
                content=full_sec_text,
                token_count=len(words),
            )
            parents.append(parent)

            # Generate Child Chunks with sliding window of child_size and child_overlap
            step = max(1, self.child_size - self.child_overlap)
            for i in range(0, len(words), step):
                chunk_words = words[i : i + self.child_size]
                if not chunk_words:
                    continue
                child_text = " ".join(chunk_words)
                child = ChildChunk(
                    document_id=document.id,
                    chunk_index=len(children),
                    parent_chunk_id=parent.id,
                    content=child_text,
                    token_count=len(chunk_words),
                )
                children.append(child)
                parent.child_chunk_ids.append(child.id)

                if i + self.child_size >= len(words):
                    break

        logger.info(f"Produced {len(parents)} Parent chunks and {len(children)} Child chunks.")
        return parents, children
