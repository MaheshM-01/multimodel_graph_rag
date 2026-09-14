"""Knowledge Graph Retriever executing Local K-hop Traversal, Global Community Summaries, and Text2Cypher."""

from src.core.constants import EdgeType, ModalityType
from src.core.logging import logger
from src.domain.retrieval import QueryRequest, SearchResult
from src.retrieval.base import BaseRetriever
from src.storage.graph_store import BaseGraphStore


class GraphRetriever(BaseRetriever):
    """Executes Local K-hop entity traversal, Global community reports, and Text2Cypher aggregations."""

    def __init__(self, graph_store: BaseGraphStore):
        self.graph_store = graph_store
        logger.info("Initialized multi-mode GraphRetriever")

    async def retrieve(self, request: QueryRequest) -> list[SearchResult]:
        """Default retrieval delegating to Local, Global, or Cypher based on request metadata."""
        if not request.query_text:
            return []

        # Check if caller specified a particular graph mode
        mode = request.metadata.get("graph_mode", "local") if hasattr(request, "metadata") else "local"

        if mode == "global" or request.include_community_reports:
            # Combine Local and Global results
            local_res = await self.local_search(request.query_text, top_k=request.top_k)
            global_res = await self.global_community_search(request.query_text, top_k=5)
            return local_res + global_res

        return await self.local_search(request.query_text, top_k=request.top_k)

    async def local_search(self, query_text: str, top_k: int = 10) -> list[SearchResult]:
        """1. Local Search: Entity-guided K-hop expansion around named entities."""
        logger.debug(f"[GraphRetriever:Local] Running 2-hop neighborhood expansion for '{query_text}'")

        # Multi-hop query: Entity ➔ related Entities ➔ Chunks ➔ Images
        cypher = f"""
        MATCH (e:Entity)
        WHERE e.name =~ '(?i).*' + $keyword + '.*'
        OPTIONAL MATCH (e)-[r:{EdgeType.RELATES_TO}*1..2]-(connected:Entity)
        OPTIONAL MATCH (e)<-[:{EdgeType.MENTIONS}]-(c:Chunk)
        OPTIONAL MATCH (e)<-[:{EdgeType.CHARTED_IN}|{EdgeType.DEPICTS}]-(img:Image)
        RETURN e.name AS primary_entity,
               collect(DISTINCT connected.name)[..5] AS related_entities,
               collect(DISTINCT c.content_snippet)[..3] AS text_snippets,
               collect(DISTINCT img.image_url)[..2] AS images
        LIMIT $limit
        """
        records = await self.graph_store.query(cypher, {"keyword": query_text, "limit": top_k})
        results: list[SearchResult] = []

        for row in records:
            entity_name = row.get("primary_entity")
            related = row.get("related_entities", [])
            snippets = row.get("text_snippets", [])
            images = row.get("images", [])

            content_lines = [f"Entity: {entity_name}"]
            if related:
                content_lines.append(f"Connected Entities: {', '.join(related)}")
            if snippets:
                content_lines.append(f"Evidence Chunks: {' | '.join(snippets)}")

            results.append(
                SearchResult(
                    id=f"graph_local_{entity_name}",
                    modality=ModalityType.TEXT,
                    score=1.0,
                    content="\n".join(content_lines),
                    image_url=images[0] if images else None,
                    source_type="graph",
                    metadata={"entity": entity_name, "related": related},
                )
            )

        return results

    async def global_community_search(self, query_text: str, top_k: int = 5) -> list[SearchResult]:
        """2. Global Search: Leiden / Louvain community summaries for thematic queries."""
        logger.debug(f"[GraphRetriever:Global] Fetching community summaries for '{query_text}'")

        cypher = """
        MATCH (cm:Community)
        RETURN cm.id AS id, cm.title AS title, cm.summary AS summary, cm.findings AS findings
        LIMIT $limit
        """
        records = await self.graph_store.query(cypher, {"limit": top_k})
        results: list[SearchResult] = []

        for row in records:
            summary = row.get("summary")
            if not summary:
                continue
            cid = row.get("id", "comm_0")
            title = row.get("title", "Thematic Community")
            findings = row.get("findings", [])

            content = f"Community Report: {title}\nSummary: {summary}\nFindings: {findings}"
            results.append(
                SearchResult(
                    id=f"community_{cid}",
                    modality=ModalityType.TEXT,
                    score=0.9,
                    content=content,
                    source_type="community",
                    metadata={"community_id": cid, "title": title},
                )
            )

        return results

    async def execute_text2cypher(self, cypher_query: str) -> list[SearchResult]:
        """3. Text2Cypher: Direct execution of structured analytical queries."""
        logger.debug(f"[GraphRetriever:Cypher] Executing direct query: {cypher_query}")
        records = await self.graph_store.query(cypher_query)
        results: list[SearchResult] = []

        for idx, row in enumerate(records):
            results.append(
                SearchResult(
                    id=f"cypher_row_{idx}",
                    modality=ModalityType.TEXT,
                    score=1.0,
                    content=str(row),
                    source_type="cypher",
                    metadata=row,
                )
            )

        return results
