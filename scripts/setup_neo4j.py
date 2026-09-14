"""Script to initialize constraints and vector indexes in Neo4j."""

import asyncio
from src.core.logging import logger, setup_logging
from src.graph.client import get_graph_client
from src.graph.schema import GraphSchemaManager


async def main():
    setup_logging("INFO")
    logger.info("Initializing Neo4j Knowledge Graph Schema...")
    client = get_graph_client()
    try:
        await client.connect()
        schema_manager = GraphSchemaManager(client)
        await schema_manager.initialize_schema()
        logger.info("Schema initialized successfully!")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
