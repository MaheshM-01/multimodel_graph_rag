"""API v1 router aggregator."""

from fastapi import APIRouter
from src.api.v1.endpoints import documents, graph, health, ingest, jobs, query

api_v1_router = APIRouter()

api_v1_router.include_router(health.router)
api_v1_router.include_router(ingest.router, prefix="/ingest")
api_v1_router.include_router(jobs.router, prefix="/jobs")
api_v1_router.include_router(query.router, prefix="/rag")
api_v1_router.include_router(graph.router, prefix="/graph")
api_v1_router.include_router(documents.router, prefix="/documents")
