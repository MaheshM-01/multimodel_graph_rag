"""Document upload and asynchronous ingestion dispatching endpoints."""

from pathlib import Path
from fastapi import APIRouter, File, UploadFile, status
from pydantic import BaseModel
from src.core.logging import logger
from src.domain.jobs import IngestionJob, JobStatus
from src.worker import register_job

router = APIRouter()


class IngestionEnqueueResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    message: str
    check_status_url: str


@router.post(
    "/upload",
    response_model=IngestionEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
)
async def upload_document(
    file: UploadFile = File(...),
) -> IngestionEnqueueResponse:
    """Upload multimodal document (PDF, PNG, JPG) and enqueue for background worker processing."""
    filename = file.filename or "unknown_file"
    logger.info(f"Received file upload: {filename} (mime: {file.content_type})")

    # Ensure storage directory exists
    storage_dir = Path("./data/media_store")
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir / filename

    # Read and persist file bytes
    file_bytes = await file.read()
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved uploaded document to {file_path} ({len(file_bytes)} bytes)")

    # Create ingestion job tracking object
    job = IngestionJob(
        filename=filename,
        file_path=str(file_path),
        mime_type=file.content_type or "application/octet-stream",
        status=JobStatus.PENDING,
        current_stage="queued for worker",
        progress_percentage=10,
    )

    # Register into asynchronous worker queue and trigger processing
    register_job(job)

    return IngestionEnqueueResponse(
        job_id=job.job_id,
        filename=filename,
        status=job.status,
        message="Document enqueued for asynchronous multimodal parsing and knowledge graph extraction.",
        check_status_url=f"/api/v1/jobs/{job.job_id}",
    )
