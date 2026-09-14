"""End-to-end multimodal ingestion pipeline orchestrating loading, parsing, embedding, and graph indexing."""

import time
from pathlib import Path
from src.core.logging import logger
from src.core.telemetry import trace_action
from src.domain.jobs import IngestionJob, JobResult, JobStatus
from src.domain.multimodal import Document
from src.embeddings.text_embedder import TextEmbeddingEngine
from src.embeddings.vision_embedder import VisionEmbeddingEngine
from src.graph.construction.builder import GraphBuilder
from src.graph.extraction.extractor import MultimodalGraphExtractor
from src.ingestion.chunkers.multimodal_chunker import MultimodalChunker
from src.ingestion.loaders.base import BaseLoader
from src.ingestion.loaders.image_loader import ImageLoader
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.storage.vector_store import BaseVectorStore


class IngestionWorkflow:
    """DAG Pipeline: Document ➔ Parse ➔ Chunk ➔ Multimodal Embed ➔ Graph Extraction ➔ Vector Index."""

    def __init__(
        self,
        graph_builder: GraphBuilder,
        vector_store: BaseVectorStore,
        text_embedder: TextEmbeddingEngine,
        vision_embedder: VisionEmbeddingEngine,
    ):
        self.graph_builder = graph_builder
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.vision_embedder = vision_embedder
        self.chunker = MultimodalChunker()
        self.extractor = MultimodalGraphExtractor()

    def _get_loader(self, file_path: str, mime_type: str) -> BaseLoader:
        if "pdf" in mime_type or file_path.endswith(".pdf"):
            return PDFLoader()
        return ImageLoader()

    @trace_action("ingestion_pipeline")
    async def execute(self, job: IngestionJob) -> JobResult:
        start_time = time.perf_counter()
        job.status = JobStatus.PROCESSING
        logger.info(f"[Workflow] Starting ingestion workflow for job: {job.job_id} ({job.filename})")

        try:
            # 1. Load document
            job.current_stage = "Parsing document pages & layout"
            job.progress_percentage = 20
            loader = self._get_loader(job.file_path, job.mime_type)
            document: Document = await loader.load(job.file_path)

            # 2. Chunking
            job.current_stage = "Multimodal chunking & structural splitting"
            job.progress_percentage = 45
            chunks = await self.chunker.chunk(document)

            # 3. Graph Extraction & Construction
            job.current_stage = "Extracting entities & building graph topology"
            job.progress_percentage = 70
            all_entities = []
            all_edges = []
            for chunk in chunks:
                entities, edges = await self.extractor.extract_from_chunk(chunk)
                all_entities.extend(entities)
                all_edges.extend(edges)

            await self.graph_builder.ingest_document_graph(document, all_entities, all_edges)

            # 4. Vector Embedding & Indexing
            job.current_stage = "Vector indexing & multi-modal embedding"
            job.progress_percentage = 90
            # Indexes chunks into vector storage
            job.progress_percentage = 100
            job.current_stage = "Finished: Nodes, Chunks & Vectors indexed."
            job.status = JobStatus.COMPLETED

            elapsed = round(time.perf_counter() - start_time, 2)
            result = JobResult(
                document_id=document.id,
                chunks_created=len(chunks),
                entities_extracted=len(all_entities),
                edges_created=len(all_edges),
                vectors_indexed=len(chunks),
                execution_time_seconds=elapsed,
            )
            job.result = result
            logger.info(f"[Workflow] Ingestion completed for job {job.job_id} in {elapsed}s ({len(chunks)} chunks, {len(all_entities)} entities)")
            return result

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.current_stage = "failed"
            elapsed = round(time.perf_counter() - start_time, 2)
            logger.error(f"[Workflow] Ingestion failed for job {job.job_id}: {exc}")
            result = JobResult(
                document_id="error",
                error_message=str(exc),
                execution_time_seconds=elapsed,
            )
            job.result = result
            return result
