"""Domain models for background asynchronous ingestion tasks and job states."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Execution status of an asynchronous background job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResult(BaseModel):
    """Outcome and statistics from a completed job."""

    document_id: str
    chunks_created: int = 0
    entities_extracted: int = 0
    edges_created: int = 0
    vectors_indexed: int = 0
    error_message: str | None = None
    execution_time_seconds: float = 0.0


class IngestionJob(BaseModel):
    """Tracks a background multimodal ingestion task."""

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    file_path: str
    mime_type: str
    status: JobStatus = JobStatus.PENDING
    progress_percentage: int = 0
    current_stage: str = "queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: JobResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
