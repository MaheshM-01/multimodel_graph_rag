"""Knowledge graph constraints, indexes, and schema manager."""

from src.core.logging import logger
from src.graph.client import Neo4jClient


class GraphSchemaManager:
    """Manages creation and verification of constraints and indexes in Neo4j."""

    CONSTRAINTS = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (cm:Community) REQUIRE cm.id IS UNIQUE",
    ]

    INDEXES = [
        "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
        "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
        "CREATE INDEX IF NOT EXISTS FOR (i:Image) ON (i.media_asset_id)",
    ]

    def __init__(self, client: Neo4jClient):
        self.client = client

    async def initialize_schema(self) -> None:
        """Create all required uniqueness constraints and lookup indexes."""
        logger.info("Initializing Neo4j schema constraints and indexes...")
        for query in self.CONSTRAINTS:
            await self.client.execute_query(query)

        for query in self.INDEXES:
            await self.client.execute_query(query)
        logger.info("Neo4j schema constraints and indexes successfully initialized.")
