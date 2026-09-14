"""Background Task Worker for heavy multimodal parsing, embedding, and graph ingestion."""

import asyncio
from src.core.logging import logger, setup_logging
from src.domain.jobs import IngestionJob, JobStatus
from src.embeddings.text_embedder import TextEmbeddingEngine
from src.embeddings.vision_embedder import VisionEmbeddingEngine
from src.graph.client import get_graph_client
from src.graph.construction.builder import GraphBuilder
from src.storage.vector_store import QdrantVectorStore
from src.workflows.ingestion_pipeline import IngestionWorkflow

# In-memory / Redis task registry
JOB_REGISTRY: dict[str, IngestionJob] = {}
JOB_QUEUE: asyncio.Queue[IngestionJob] = asyncio.Queue()
_WORKER_TASK: asyncio.Task | None = None
_WORKFLOW_INSTANCE: IngestionWorkflow | None = None


def get_job(job_id: str) -> IngestionJob | None:
    """Retrieve job state by job_id."""
    return JOB_REGISTRY.get(job_id)


def get_or_create_workflow() -> IngestionWorkflow:
    """Retrieve or initialize the singleton IngestionWorkflow."""
    global _WORKFLOW_INSTANCE
    if _WORKFLOW_INSTANCE is None:
        graph_client = get_graph_client()
        _WORKFLOW_INSTANCE = IngestionWorkflow(
            graph_builder=GraphBuilder(graph_client),
            vector_store=QdrantVectorStore(),
            text_embedder=TextEmbeddingEngine(),
            vision_embedder=VisionEmbeddingEngine(),
        )
    return _WORKFLOW_INSTANCE


def register_job(job: IngestionJob) -> None:
    """Register job in the registry and queue, ensuring worker is active."""
    JOB_REGISTRY[job.job_id] = job
    JOB_QUEUE.put_nowait(job)
    logger.info(f"Enqueued job {job.job_id} for file {job.filename}")
    ensure_worker_running()


def ensure_worker_running() -> asyncio.Task | None:
    """Ensure that the background worker loop is running inside the active event loop."""
    global _WORKER_TASK
    try:
        loop = asyncio.get_running_loop()
        if _WORKER_TASK is None or _WORKER_TASK.done():
            _WORKER_TASK = loop.create_task(run_worker())
            logger.info("Background ingestion task worker loop spawned.")
        return _WORKER_TASK
    except RuntimeError:
        return None


async def run_worker():
    """Background worker daemon continuously polling and processing queued ingestion tasks."""
    logger.info("Starting Multimodal Graph RAG Task Worker...")

    graph_client = get_graph_client()
    try:
        await graph_client.connect()
    except Exception as exc:
        logger.warning(f"Worker could not connect to Neo4j at startup: {exc}")

    workflow = get_or_create_workflow()
    logger.info("Worker ready to process background multimodal jobs.")

    while True:
        try:
            job = await JOB_QUEUE.get()
            logger.info(f"Worker picked up job: {job.job_id} ({job.filename})")
            await workflow.execute(job)
            JOB_QUEUE.task_done()
        except asyncio.CancelledError:
            logger.info("Worker shutdown received.")
            break
        except Exception as exc:
            logger.error(f"Worker encountered unexpected error: {exc}")


if __name__ == "__main__":
    try:
        setup_logging("INFO")
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")
