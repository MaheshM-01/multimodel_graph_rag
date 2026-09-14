"""Unit tests for background ingestion jobs and statuses."""

from src.domain.jobs import IngestionJob, JobResult, JobStatus


def test_ingestion_job_lifecycle():
    job = IngestionJob(
        filename="quarterly_report.pdf",
        file_path="/media/quarterly_report.pdf",
        mime_type="application/pdf",
    )
    assert job.status == JobStatus.PENDING
    assert job.progress_percentage == 0
    assert job.job_id is not None

    # Simulate completion
    job.status = JobStatus.COMPLETED
    job.progress_percentage = 100
    job.result = JobResult(
        document_id="doc_123",
        chunks_created=15,
        entities_extracted=40,
        edges_created=32,
        vectors_indexed=15,
        execution_time_seconds=3.45,
    )

    assert job.status == JobStatus.COMPLETED
    assert job.result.entities_extracted == 40
