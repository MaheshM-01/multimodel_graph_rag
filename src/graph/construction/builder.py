"""Graph builder responsible for inserting documents, hierarchical chunks, entities, and deduplicating."""

from src.core.constants import EdgeType, NodeType
from src.core.logging import logger
from src.domain.graph import EntityNode, GraphEdge
from src.domain.multimodal import ChildChunk, Document, ImageChunk, ParentChunk
from src.graph.client import Neo4jClient


class GraphBuilder:
    """Constructs knowledge graph topology in Neo4j with hierarchical chunks, visual entity grounding, and deduplication."""

    def __init__(self, client: Neo4jClient):
        self.client = client

    async def ingest_document_graph(
        self,
        document: Document,
        entities: list[EntityNode],
        edges: list[GraphEdge],
    ) -> None:
        """Insert Document node, Hierarchical Chunks, Entities, and connecting Relationships."""
        logger.info(
            f"Building graph for document '{document.title}' ({len(entities)} entities, {len(edges)} edges, {len(document.parent_chunks)} parent chunks)"
        )

        # 1. Merge Document Node
        merge_doc_query = """
        MERGE (d:Document {id: $id})
        ON CREATE SET d.title = $title, d.source_uri = $source_uri, d.created_at = datetime()
        ON MATCH SET d.title = $title
        """
        await self.client.execute_query(
            merge_doc_query,
            {"id": document.id, "title": document.title, "source_uri": document.source_uri},
        )

        # 2. Merge Hierarchical Parent Chunks & Link to Document (:Document)-[:HAS_SECTION]->(:ParentChunk)
        for parent in document.parent_chunks:
            merge_parent_query = f"""
            MERGE (p:{NodeType.PARENT_CHUNK} {{id: $parent_id}})
            ON CREATE SET p.section_title = $section_title, p.content = $content, p.document_id = $doc_id
            WITH p
            MATCH (d:Document {{id: $doc_id}})
            MERGE (d)-[:{EdgeType.HAS_SECTION}]->(p)
            """
            await self.client.execute_query(
                merge_parent_query,
                {
                    "parent_id": parent.id,
                    "section_title": parent.section_title,
                    "content": parent.content,
                    "doc_id": document.id,
                },
            )

        # 3. Merge Child Chunks & Link (:ChildChunk)-[:BELONGS_TO_PARENT]->(:ParentChunk)
        for chunk in document.chunks:
            if isinstance(chunk, ChildChunk):
                merge_child_query = f"""
                MERGE (c:{NodeType.CHILD_CHUNK} {{id: $child_id}})
                ON CREATE SET c.content = $content, c.contextual_header = $header, c.document_id = $doc_id
                WITH c
                MATCH (p:{NodeType.PARENT_CHUNK} {{id: $parent_id}})
                MERGE (c)-[:{EdgeType.BELONGS_TO_PARENT}]->(p)
                """
                await self.client.execute_query(
                    merge_child_query,
                    {
                        "child_id": chunk.id,
                        "content": chunk.content,
                        "header": chunk.contextual_header or "",
                        "doc_id": document.id,
                        "parent_id": chunk.parent_chunk_id,
                    },
                )

        # 4. Merge Entities
        for entity in entities:
            merge_entity_query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET e.entity_type = $entity_type, e.description = $description
            """
            await self.client.execute_query(
                merge_entity_query,
                {
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                },
            )

        # 5. Create Visual Entity Grounding Links (VISUALLY_REPRESENTS & CHARTED_IN)
        for chunk in document.chunks:
            if isinstance(chunk, ImageChunk):
                for v_link in chunk.visual_entity_links:
                    link_query = f"""
                    MATCH (img:Image {{id: $img_id}})
                    MATCH (e:Entity {{name: $entity_name}})
                    MERGE (img)-[r:{EdgeType.VISUALLY_REPRESENTS} {{
                        confidence: $confidence,
                        x_min: $x_min,
                        y_min: $y_min,
                        x_max: $x_max,
                        y_max: $y_max
                    }}]->(e)
                    MERGE (e)-[:{EdgeType.CHARTED_IN}]->(img)
                    """
                    await self.client.execute_query(
                        link_query,
                        {
                            "img_id": chunk.id,
                            "entity_name": v_link.entity_name,
                            "confidence": v_link.confidence,
                            "x_min": v_link.bounding_box.x_min,
                            "y_min": v_link.bounding_box.y_min,
                            "x_max": v_link.bounding_box.x_max,
                            "y_max": v_link.bounding_box.y_max,
                        },
                    )

        logger.info(f"Successfully committed hierarchical graph nodes and visual links for document {document.id}")
