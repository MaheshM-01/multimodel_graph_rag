"""FastAPI application factory, static files mounting, and lifespan configuration."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.api.v1.router import api_v1_router
from src.config.settings import Settings, get_settings
from src.core.logging import logger, setup_logging
from src.graph.client import get_graph_client

import asyncio
from src.worker import run_worker

STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown events."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger.info(f"Starting {settings.APP_NAME} in [{settings.APP_ENV}] mode...")

    # Initialize Graph Database Client
    graph_client = get_graph_client()
    try:
        await graph_client.connect()
    except Exception as exc:
        logger.warning(f"Neo4j connection could not be established at startup: {exc}")

    # Start the continuous background ingestion worker
    worker_task = asyncio.create_task(run_worker())
    logger.info("Background ingestion task worker successfully started.")

    yield

    logger.info("Shutting down application...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    await graph_client.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure production FastAPI application."""
    current_settings = settings or get_settings()

    app = FastAPI(
        title=current_settings.APP_NAME,
        description="Production-grade Multimodal Graph RAG engine uniting Knowledge Graphs, Vector Search, and Vision LLMs.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if current_settings.DEBUG or current_settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if current_settings.DEBUG or current_settings.APP_ENV != "production" else None,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routers
    app.include_router(api_v1_router, prefix=current_settings.API_V1_PREFIX)

    # Mount Static Files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Serve Main Frontend Dashboard at Root /
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {
            "status": "online",
            "message": "Multimodal Graph RAG API is operational. Static UI is initializing.",
            "api_docs": "/docs",
        }

    return app
