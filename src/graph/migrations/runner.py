"""Cypher database migration runner."""

from pathlib import Path
from src.core.logging import logger
from src.graph.client import Neo4jClient


class MigrationRunner:
    """Executes declarative .cypher migration scripts against Neo4j."""

    def __init__(self, client: Neo4jClient, migrations_dir: str | Path | None = None):
        self.client = client
        self.migrations_dir = Path(migrations_dir or Path(__file__).parent)

    async def run_migrations(self) -> list[str]:
        """Execute all pending .cypher migration scripts in alphabetical order."""
        executed: list[str] = []
        cypher_files = sorted(self.migrations_dir.glob("*.cypher"))

        for file in cypher_files:
            logger.info(f"Running Cypher migration: {file.name}")
            content = file.read_text(encoding="utf-8")
            statements = [s.strip() for s in content.split(";") if s.strip() and not s.strip().startswith("//")]

            for stmt in statements:
                await self.client.execute_query(stmt)

            executed.append(file.name)
            logger.info(f"Successfully applied migration: {file.name}")

        return executed
