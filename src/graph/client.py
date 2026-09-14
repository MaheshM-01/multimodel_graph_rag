"""Neo4j graph database client and connection pool management."""

from typing import Any
from src.config.settings import Settings, get_settings
from src.core.exceptions import GraphDatabaseError
from src.core.logging import logger

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None  # type: ignore
    AsyncDriver = Any  # type: ignore


class Neo4jClient:
    """Async Neo4j driver connection pool wrapper."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize the connection pool to Neo4j."""
        if not NEO4J_AVAILABLE:
            logger.warning("neo4j package not installed. Operating in offline/mock graph mode.")
            return

        try:
            logger.info(f"Connecting to Neo4j at {self.settings.NEO4J_URI}...")
            self._driver = AsyncGraphDatabase.driver(
                self.settings.NEO4J_URI,
                auth=(self.settings.NEO4J_USER, self.settings.NEO4J_PASSWORD),
                max_connection_pool_size=self.settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
            )
            await self._driver.verify_connectivity()
            logger.info("Successfully established connection to Neo4j.")
        except Exception as exc:
            logger.error(f"Failed to connect to Neo4j: {exc}")
            raise GraphDatabaseError(f"Cannot connect to Neo4j: {exc}") from exc

    async def close(self) -> None:
        """Close driver connection pool."""
        if self._driver:
            await self._driver.close()
            logger.info("Closed Neo4j connection pool.")

    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return records as dictionaries."""
        if not self._driver:
            logger.warning("Neo4j driver not connected. Mock query returning empty list.")
            return []

        try:
            async with self._driver.session(database=self.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
        except Exception as exc:
            logger.error(f"Cypher execution failed for query: '{query[:100]}...' Error: {exc}")
            raise GraphDatabaseError(f"Query execution error: {exc}") from exc


_graph_client: Neo4jClient | None = None


def get_graph_client() -> Neo4jClient:
    """Singleton getter for Neo4j client."""
    global _graph_client
    if _graph_client is None:
        _graph_client = Neo4jClient()
    return _graph_client
