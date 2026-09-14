"""Multimodal RAG synthesizer assembling structured Graph, Text, and Visual context with citations."""

import time
from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.generation import GroundingMetadata, MultimodalCitation, RAGRequest, RAGResponse
from src.domain.retrieval import SearchResult
from src.generation.llm_client import LLMClient
from src.generation.prompts.system_prompts import GRAPH_RAG_SYSTEM_PROMPT


class MultimodalSynthesizer:
    """Assembles retrieved Quad-Hybrid multimodal context into structured sections:

    1. Knowledge Graph Subgraph (Structured Facts & Relations)
    2. Highly Relevant Text Excerpts
    3. Visual & Table Grounding (Charts, Bounding Boxes, Extracted Data Points)
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def synthesize(
        self, request: RAGRequest, retrieved_items: list[SearchResult]
    ) -> RAGResponse:
        start_time = time.perf_counter()
        logger.info(f"Synthesizing response for query: '{request.query}' with {len(retrieved_items)} context items")

        # Segregate into 3 Structured Context Buckets
        graph_facts: list[str] = []
        text_excerpts: list[str] = []
        visual_groundings: list[str] = []
        citations: list[MultimodalCitation] = []
        graph_entities: list[str] = []

        for idx, item in enumerate(retrieved_items, start=1):
            source_tag = f"[{idx}]"

            # 1. Knowledge Graph items
            if item.source_type in ("graph", "community", "cypher"):
                graph_facts.append(f"{source_tag} {item.content}")
                if entity := item.metadata.get("entity"):
                    graph_entities.append(entity)

            # 2. Visual & Table Groundings (for items with diagrams or image modality)
            if item.modality == ModalityType.IMAGE or item.source_type == "visual":
                datapoints_str = f" | Data: {item.data_points}" if item.data_points else ""
                visual_groundings.append(
                    f"{source_tag} [Visual Asset ID: {item.id}] Content: {item.content}{datapoints_str} (URL: {item.image_url or 'inline'})"
                )

            # 3. Text Chunks & Document Page Excerpts
            if item.content and item.source_type not in ("graph", "community", "cypher"):
                text_excerpts.append(f"{source_tag} {item.content}")

            # Register citation
            citations.append(
                MultimodalCitation(
                    citation_index=idx,
                    source_chunk_id=item.id,
                    document_id=item.metadata.get("document_id", "unknown"),
                    page_number=item.metadata.get("page_number"),
                    media_url=item.image_url,
                    snippet=item.content[:150],
                    modality=str(item.modality),
                )
            )

        # Build Structured Context for LLM
        context_parts = ["# CONTEXT FOR ANSWERING THE QUERY:\n"]

        if graph_facts:
            context_parts.append("## 1. Knowledge Graph Subgraph (Structured Facts):\n" + "\n".join(graph_facts))

        if text_excerpts:
            context_parts.append("\n## 2. Highly Relevant Text Excerpts:\n" + "\n".join(text_excerpts))

        if visual_groundings:
            context_parts.append("\n## 3. Visual & Table Grounding:\n" + "\n".join(visual_groundings))

        full_context = "\n".join(context_parts)
        full_prompt = f"User Question: {request.query}\n\n{full_context}\n\nSynthesized Grounded Answer:"

        # Call LLM
        answer = await self.llm_client.generate_text(
            prompt=full_prompt,
            system_prompt=GRAPH_RAG_SYSTEM_PROMPT,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        latency = round(time.perf_counter() - start_time, 3)

        return RAGResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            grounding=GroundingMetadata(
                retrieved_chunk_count=len(retrieved_items),
                graph_entities_used=list(set(graph_entities)),
            ),
            latency_seconds=latency,
        )
