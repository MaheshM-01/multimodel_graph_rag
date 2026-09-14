# 🚀 Multimodal Graph RAG System (Enterprise Production Grade)

A production-grade, enterprise-ready **Multimodal Graph Retrieval-Augmented Generation (Graph RAG)** platform with a stunning, modern dark-mode web application dashboard. This architecture seamlessly combines **Knowledge Graph Traversal (Neo4j)**, **Dense Vector Retrieval (Qdrant)**, **Sparse BM25 Keyword Search**, and **Vision-Language Models (VLMs)** to provide hallucination-resistant, fully grounded reasoning over complex multimodal data (documents, charts, diagrams, tables, and images).

---

## 🎨 Interactive Frontend Dashboard (`http://localhost:8000/`)

The application includes a built-in, luxury dark-theme glassmorphic single-page web dashboard served directly by FastAPI:
- **💬 Multimodal RAG Assistant**: Interactive chat accepting queries with image/chart attachments, real-time response rendering, and clickable citation pills.
- **🕸️ Knowledge Graph Visualizer**: Interactive 2D force-directed physics network canvas powered by **Vis-Network**. Explore `Entity`, `Image`, `Chunk`, and `Community` nodes with clickable relationship inspection (`VISUALLY_REPRESENTS`, `CHARTED_IN`, `RELATES_TO`).
- **📥 Document Ingestion & Job Center**: Drag-and-drop file upload for PDFs and images with animated progress bars tracking: `Chunking ➔ Embedding ➔ Graph Building ➔ Vector Indexing`.
- **📊 Provenance & Grounding Drawer**: Real-time display of Query Router intent (`FACTUAL_ENTITY`, `VISUAL_COMPARATIVE`, `GLOBAL_COMMUNITY`, `ANALYTICAL_CYPHER`), channel weights, extracted entities, and latency.

---

## 🏛️ System Architecture

```
                                  [ User Web UI / API ]
                                            │
                                            ▼
                           [ Asynchronous Task Queue / Redis ]
                                            │
                                            ▼
                               [ Background Worker (worker.py) ]
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
     [ Text & Table Chunks ]                                   [ Visual Media Chunks ]
               │                                                         │
        ┌──────┴──────┐                                           ┌──────┴──────┐
        ▼             ▼                                           ▼             ▼
  [ Text Embedder ] [ Entity/Relation Extractor ]         [ Vision Embedder ] [ Visual Entity Extractor ]
        │                     │                                   │                     │
        ▼                     ▼                                   ▼                     ▼
 ┌──────────────┐     ┌──────────────┐                     ┌──────────────┐     ┌────────────────┐
 │  Vector DB   │     │ Knowledge DB │                     │  Vector DB   │     │  Knowledge DB  │
 │ (Dense + BM25│     │ (Entities &  │                     │(Image Index) │     │ (Visual Nodes  │
 │   Index)     │     │ Relations)   │                     │              │     │  & DEPICTS)    │
 └──────┬───────┘     └──────┬───────┘                     └──────┬───────┘     └───────┬────────┘
        │                    │                                    │                     │
        └────────────────────┼────────────────────────────────────┴─────────────────────┘
                             ▼
                [ Quad-Hybrid Fusion Retriever ]
            (Dense Vector + Sparse BM25 + Visual ColPali + Graph Cypher via Weighted RRF)
                             │
                             ▼
               [ 2-Stage Multimodal Reranker ]
            (Stage 1: Text Cross-Encoder + Stage 2: Visual Grounding Check)
                             │
                             ▼
             [ Structured Multimodal Synthesizer ]
              (LLM / VLM + Grounded Citations)
```

---

## 📂 Production Folder Structure

```
multimodel_rag_graph_search/
├── .env.example                     # Production environment blueprint (Neo4j, Qdrant, MinIO, Redis, LLMs)
├── docker-compose.yml               # Orchestration for Neo4j (APOC), Qdrant, MinIO, Redis, Worker, & API
├── Dockerfile                       # Multi-stage production container build with OCR support
├── Makefile                         # Automation commands (make dev, make test, make docker-up, etc.)
├── pyproject.toml                   # Modern dependencies & Ruff / Pytest configurations
├── README.md                        # Complete architecture docs, workflow, & quickstart
│
├── data/
│   └── media_store/                 # Blob storage for extracted images, charts & PDFs
│
├── scripts/
│   ├── setup_neo4j.py               # Auto-create uniqueness constraints & indexes
│   └── demo_ingest.py               # Sample ingestion & graph construction script
│
├── src/
│   ├── __init__.py
│   ├── main.py                      # ASGI entrypoint & uvicorn runner
│   ├── worker.py                    # Async background worker entrypoint
│   │
│   ├── api/                         # FastAPI Web API
│   │   ├── app.py                   # App factory, static files mounting, lifespan & CORS
│   │   ├── dependencies.py          # Dependency injection (DB pools, clients, engines)
│   │   └── v1/
│   │       ├── router.py            # API v1 route aggregator
│   │       └── endpoints/
│   │           ├── health.py        # Liveness & readiness probes
│   │           ├── ingest.py        # Async file upload (returns job_id, HTTP 202)
│   │           ├── jobs.py          # Polling endpoint for ingestion job status
│   │           ├── query.py         # Multimodal Graph RAG chat endpoint
│   │           └── graph.py         # Subgraph & visualizer canvas overview endpoints
│   │
│   ├── static/                      # Modern Web Application Dashboard (SPA)
│   │   ├── index.html               # Main dashboard UI (Chat, Visualizer, Ingest, Telemetry)
│   │   ├── css/
│   │   │   └── style.css            # Dark-mode glassmorphic design system (CSS custom properties)
│   │   └── js/
│   │       ├── app.js               # Global state, tab routing & health probes
│   │       ├── chat.js              # Chat interface, citations, modal viewer & provenance drawer
│   │       ├── graph_view.js        # Vis-Network 2D force-directed interactive graph canvas
│   │       └── ingest_view.js       # Drag-and-drop upload & live job status progress bar
│   │
│   ├── config/
│   │   └── settings.py              # Pydantic v2 Settings with `.env` overrides
│   │
│   ├── core/                        # Core system utilities
│   │   ├── constants.py             # Enums: ModalityType, NodeType, EdgeType
│   │   ├── exceptions.py            # Domain-specific typed exceptions
│   │   ├── logging.py               # Structured logging (Loguru / standard logging)
│   │   ├── security.py              # API Key authentication
│   │   └── telemetry.py             # OpenTelemetry / Langfuse distributed tracing
│   │
│   ├── domain/                      # Domain-Driven Design (DDD) Pydantic schemas
│   │   ├── multimodal.py            # Document, ParentChunk, ChildChunk, PropositionChunk, ImageChunk
│   │   ├── graph.py                 # EntityNode, ImageNode, ChunkNode, GraphEdge, SubGraph
│   │   ├── retrieval.py             # QueryIntent, RouterDecision, QueryRequest, SearchResult, FusionItem
│   │   ├── generation.py            # RAGRequest, RAGResponse, MultimodalCitation
│   │   └── jobs.py                  # IngestionJob, JobStatus, JobResult
│   │
│   ├── workflows/                   # Workflow DAG Orchestration Layer
│   │   ├── ingestion_pipeline.py    # Ingest ➔ Parse ➔ Embed ➔ Build Graph ➔ Vector Index
│   │   └── rag_pipeline.py          # Query ➔ Router ➔ Quad-Hybrid Search ➔ Weighted RRF ➔ 2-Stage Rerank
│   │
│   ├── ingestion/                   # Top 6 Advanced Chunking & Extraction
│   │   ├── loaders/                 # PDFLoader, ImageLoader
│   │   ├── parsers/                 # LayoutParser, OCRParser, TableParser
│   │   └── chunkers/                # LayoutAware, Contextual, Hierarchical, CrossModal, Propositional
│   │
│   ├── embeddings/                  # Embedding engines
│   │   ├── text_embedder.py         # Dense text embeddings
│   │   ├── vision_embedder.py       # Vision embeddings (CLIP/SigLIP/ColPali)
│   │   └── fusion.py                # Cross-modal projection & fusion
│   │
│   ├── graph/                       # Knowledge Graph engine
│   │   ├── client.py                # Async Neo4j connection pool
│   │   ├── schema.py                # Constraints & index management
│   │   ├── migrations/              # Cypher schema migrations & runner
│   │   ├── extraction/              # LLM/VLM entity-relation extractor
│   │   ├── construction/            # GraphBuilder (insert nodes, deduplicate, visual links)
│   │   └── community/               # GraphRAG community detection (Leiden/Louvain)
│   │
│   ├── storage/                     # Persistence adapters
│   │   ├── graph_store.py           # Neo4j graph store adapter
│   │   ├── vector_store.py          # Qdrant vector store adapter
│   │   ├── media_store.py           # Local / MinIO / S3 blob store
│   │   └── cache.py                 # Redis / Memory query cache adapter
│   │
│   ├── retrieval/                   # Quad-Hybrid retrieval engine
│   │   ├── query_router.py          # Smart intent classification
│   │   ├── vector_search.py         # Channel 1: Dense vector search
│   │   ├── sparse_search.py         # Channel 2: BM25 keyword search
│   │   ├── visual_search.py         # Channel 3: ColPali/SigLIP visual search
│   │   ├── graph_search.py          # Channel 4: Local K-hop, Global, Text2Cypher
│   │   ├── hybrid_fusion.py         # Weighted Reciprocal Rank Fusion (Weighted RRF)
│   │   └── reranker.py              # 2-Stage Multimodal Reranker
│   │
│   ├── generation/                  # Answer synthesis
│   │   ├── llm_client.py            # Multi-provider LLM gateway
│   │   ├── synthesizer.py           # Grounded response synthesis with 3-section structured context
│   │   └── prompts/                 # System prompts for GraphRAG & citations
│   │
│   └── evaluation/                  # Observability & evaluation
│       ├── metrics.py               # Groundedness & citation precision
│       └── benchmarks.py            # Automated test benchmark runner
│
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_domain_models.py
    │   ├── test_hybrid_fusion.py
    │   ├── test_jobs.py
    │   ├── test_sparse_search.py
    │   ├── test_workflows.py
    │   ├── test_advanced_chunkers.py
    │   └── test_quad_retrieval.py
    └── integration/
        ├── test_api_endpoints.py
        └── test_frontend_routes.py
```

---

## ⚡ Quick Start

### 1. Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 2. Start Full Infrastructure Stack (Docker)

Spin up Neo4j, Qdrant, MinIO, Redis, and Worker:
```bash
docker-compose up -d
```
- **Neo4j Browser**: [http://localhost:7474](http://localhost:7474) (`neo4j` / `password123`)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **MinIO Console**: [http://localhost:9001](http://localhost:9001) (`minioadmin` / `minioadmin`)

### 3. Run FastAPI Application & Web UI

```bash
make dev
# or
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
- **Web UI Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run Test Suite

```bash
make test
# or
pytest tests/
```
