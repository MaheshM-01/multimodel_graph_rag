"""Production-grade Text Embedding Engine powered by SentenceTransformers.
Generates normalized 384-dimensional dense semantic vector representations for text chunks and queries.
"""

from typing import List, Optional
import numpy as np
from src.config.settings import get_settings
from src.core.logging import logger
from src.embeddings.base import BaseEmbeddingEngine

# Lazy-loaded model singleton to prevent redundant model weight loads across threads
_GLOBAL_EMBEDDING_MODEL = None
_GLOBAL_MODEL_NAME = None


def _get_sentence_transformer(model_name: str):
    global _GLOBAL_EMBEDDING_MODEL, _GLOBAL_MODEL_NAME
    if _GLOBAL_EMBEDDING_MODEL is None or _GLOBAL_MODEL_NAME != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer embedding model: '{model_name}'...")
            _GLOBAL_EMBEDDING_MODEL = SentenceTransformer(model_name)
            _GLOBAL_MODEL_NAME = model_name
            logger.info(f"SentenceTransformer '{model_name}' successfully loaded into memory.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer '{model_name}': {e}. Using fallback embedding.")
            _GLOBAL_EMBEDDING_MODEL = None
    return _GLOBAL_EMBEDDING_MODEL


class TextEmbeddingEngine(BaseEmbeddingEngine):
    """Generates dense normalized semantic vector embeddings using SentenceTransformers."""

    def __init__(self, model_name: Optional[str] = None, dim: int = 384):
        settings = get_settings()
        # Default to high-performance, proven all-MiniLM-L6-v2
        self.model_name = model_name or getattr(settings, "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.dim = dim
        self._cache: dict[str, list[float]] = {}

    def _get_model(self):
        return _get_sentence_transformer(self.model_name)

    def _fallback_embedding(self, text: str) -> list[float]:
        """Deterministic subword n-gram embedding fallback if transformer is unavailable."""
        import hashlib
        import re
        tokens = re.findall(r"\b\w+\b", text.lower().strip())
        if not tokens:
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = 1.0
            return v.tolist()

        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        else:
            vec[0] = 1.0
        return [round(float(x), 6) for x in vec]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into a (N, dim) normalized float32 numpy matrix."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        model = self._get_model()
        if model is not None:
            try:
                embeddings = model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=64
                )
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.error(f"Transformer batch encoding error: {e}. Falling back.")

        # Fallback encoding
        vectors = [self._fallback_embedding(t) for t in texts]
        return np.array(vectors, dtype=np.float32)

    def encode_text(self, text: str) -> list[float]:
        """Encode a single text string into a normalized dense vector."""
        if text in self._cache:
            return self._cache[text]

        model = self._get_model()
        if model is not None:
            try:
                vec = model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                res = [round(float(x), 6) for x in vec]
                self._cache[text] = res
                return res
            except Exception as e:
                logger.error(f"Transformer text encoding error: {e}")

        res = self._fallback_embedding(text)
        self._cache[text] = res
        return res

    async def get_embedding(self, input_data: str | bytes) -> list[float]:
        text = input_data if isinstance(input_data, str) else input_data.decode("utf-8", errors="ignore")
        return self.encode_text(text)

    async def get_batch_embeddings(self, inputs: list[str | bytes]) -> list[list[float]]:
        texts = [i if isinstance(i, str) else i.decode("utf-8", errors="ignore") for i in inputs]
        matrix = self.encode_batch(texts)
        return [[round(float(x), 6) for x in row] for row in matrix]
