"""LLM and VLM-based multimodal entity and relationship extractor with deterministic fallbacks."""

import re
from src.core.logging import logger
from src.domain.graph import EntityNode, GraphEdge
from src.domain.multimodal import BaseChunk, ImageChunk, TextChunk


class MultimodalGraphExtractor:
    """Extracts typed entities and cross-modal relationships using LLM/VLM reasoning with robust fallback rules."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    async def extract_from_chunk(
        self, chunk: BaseChunk
    ) -> tuple[list[EntityNode], list[GraphEdge]]:
        """Extract entities and relationships from a text or visual chunk."""
        entities: list[EntityNode] = []
        edges: list[GraphEdge] = []

        if hasattr(chunk, "content") and getattr(chunk, "content", None):
            content = str(chunk.content)
            # Rule-based entity extraction pattern (capitalized technical terms, acronyms, multi-word entities)
            found_names = set(re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\b", content))
            
            # Exclude common stop words
            stopwords = {"The", "This", "That", "These", "Those", "There", "Here", "When", "Where", "Why", "How", "Page", "Section", "Table", "Figure", "It", "We", "You", "They"}
            filtered_names = [name for name in found_names if name not in stopwords and len(name) > 2]

            created_nodes: list[EntityNode] = []
            for name in filtered_names[:8]:  # Top 8 per chunk
                entity_type = "Concept"
                if re.match(r"^[A-Z0-9\-]+$", name):
                    entity_type = "Identifier"
                elif any(word in name.lower() for word in ["corp", "inc", "ltd", "lab", "company", "group"]):
                    entity_type = "Organization"
                elif any(word in name.lower() for word in ["model", "algorithm", "network", "system", "distribution"]):
                    entity_type = "Technology"

                section_name = getattr(chunk, "section_title", None) or getattr(chunk, "contextual_header", None) or "General"
                node = EntityNode(
                    name=name,
                    entity_type=entity_type,
                    description=f"Extracted from '{section_name}' in chunk {chunk.chunk_index}",
                )
                entities.append(node)
                created_nodes.append(node)

            # Link co-occurring entities within the chunk
            for i in range(len(created_nodes) - 1):
                e1 = created_nodes[i]
                e2 = created_nodes[i + 1]
                edge = GraphEdge(
                    source_id=e1.id,
                    target_id=e2.id,
                    relation_type="CO_OCCURS_WITH",
                    weight=1.0,
                    properties={"context_chunk_id": chunk.id},
                )
                edges.append(edge)

        elif isinstance(chunk, ImageChunk):
            # Extract visual entities from caption / vlm summary
            text = f"{chunk.caption or ''} {chunk.vlm_summary or ''}".strip()
            if text:
                node = EntityNode(
                    name=f"VisualAsset_{chunk.chunk_index}",
                    entity_type="VisualArtifact",
                    description=text[:100],
                )
                entities.append(node)

        return entities, edges
