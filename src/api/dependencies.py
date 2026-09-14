"""FastAPI Dependency Injection providers for database connections and engines."""

from functools import lru_cache
from src.embeddings.text_embedder import TextEmbeddingEngine
from src.embeddings.vision_embedder import VisionEmbeddingEngine
from src.generation.llm_client import LLMClient
from src.generation.synthesizer import MultimodalSynthesizer
from src.graph.client import get_graph_client
from src.retrieval.graph_search import GraphRetriever
from src.retrieval.hybrid_fusion import HybridRetriever
from src.retrieval.query_router import QueryRouter
from src.retrieval.reranker import TwoStageMultimodalReranker
from src.retrieval.sparse_search import BM25Retriever
from src.retrieval.vector_search import VectorRetriever
from src.retrieval.visual_search import VisualRetriever
from src.storage.graph_store import Neo4jGraphStore
from src.storage.vector_store import QdrantVectorStore


@lru_cache()
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()


@lru_cache()
def get_graph_store() -> Neo4jGraphStore:
    return Neo4jGraphStore(get_graph_client())


@lru_cache()
def get_text_embedder() -> TextEmbeddingEngine:
    return TextEmbeddingEngine()


@lru_cache()
def get_vision_embedder() -> VisionEmbeddingEngine:
    return VisionEmbeddingEngine()


@lru_cache()
def get_sparse_retriever() -> BM25Retriever:
    return BM25Retriever()


@lru_cache()
def get_visual_retriever() -> VisualRetriever:
    return VisualRetriever(
        vector_store=get_vector_store(),
        vision_embedder=get_vision_embedder(),
    )


@lru_cache()
def get_query_router() -> QueryRouter:
    return QueryRouter()


@lru_cache()
def get_llm_client() -> LLMClient:
    return LLMClient()


def get_hybrid_retriever() -> HybridRetriever:
    vector_retriever = VectorRetriever(
        vector_store=get_vector_store(),
        text_embedder=get_text_embedder(),
        vision_embedder=get_vision_embedder(),
    )
    graph_retriever = GraphRetriever(graph_store=get_graph_store())
    sparse_retriever = get_sparse_retriever()
    visual_retriever = get_visual_retriever()

    return HybridRetriever(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        sparse_retriever=sparse_retriever,
        visual_retriever=visual_retriever,
    )


def get_synthesizer() -> MultimodalSynthesizer:
    return MultimodalSynthesizer(llm_client=get_llm_client())


def get_reranker() -> TwoStageMultimodalReranker:
    return TwoStageMultimodalReranker()
