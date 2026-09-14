"""Endpoints for polling and tracking asynchronous background ingestion tasks."""

from fastapi import APIRouter, HTTPException, status
from src.domain.jobs import IngestionJob
from src.worker import get_job

router = APIRouter()


@router.get("/{job_id}", response_model=IngestionJob, tags=["Jobs"])
async def get_job_status(job_id: str) -> IngestionJob:
    """Retrieve current execution progress, status, and result of an ingestion job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found.",
        )
    return job
