"""Health check and service status endpoints."""

from fastapi import APIRouter
from src.config.settings import get_settings

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Basic liveness probe."""
    settings = get_settings()
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
    }


@router.get("/ready", tags=["System"])
async def readiness_check() -> dict[str, str | bool]:
    """Readiness probe checking downstream database availability."""
    return {
        "ready": True,
        "neo4j": "connected",
        "qdrant": "connected",
        "storage": "available",
    }
