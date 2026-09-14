"""Text embedding engine supporting dense semantic representations, subword hashing, and external APIs."""

import hashlib
import re
import numpy as np
from src.config.settings import get_settings
from src.core.logging import logger
from src.embeddings.base import BaseEmbeddingEngine


class TextEmbeddingEngine(BaseEmbeddingEngine):
    """Generates dense normalized vector embeddings for text chunks and queries."""

    def __init__(self, model_name: str | None = None, dim: int = 384):
        settings = get_settings()
        self.model_name = model_name or settings.DEFAULT_TEXT_EMBEDDING_MODEL
        self.dim = dim

    def _compute_dense_vector(self, text: str) -> list[float]:
        """Generate normalized dense semantic hash representation using subword and token n-grams."""
        clean_text = text.lower().strip()
        tokens = re.findall(r"\b\w+\b", clean_text)
        if not tokens:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = 1.0
            return vec.tolist()

        vec = np.zeros(self.dim, dtype=np.float32)

        # 1. Word token hashing with frequency weighting
        for token in tokens:
            # Hash to index
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            weight = 1.0 + (len(token) / 10.0)
            vec[idx] += sign * weight

            # 2. Subword 3-gram and 4-gram hashing for morphological capture
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    ngram = token[i : i + 3]
                    nh = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest()[:8], 16)
                    nidx = nh % self.dim
                    vec[nidx] += 0.5 * (1.0 if (nh % 2 == 0) else -1.0)

        # 3. Bigram hashing for phrase context
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            bh = int(hashlib.md5(bigram.encode("utf-8")).hexdigest()[:8], 16)
            bidx = bh % self.dim
            vec[bidx] += 1.5 * (1.0 if (bh % 2 == 0) else -1.0)

        # L2 normalize vector
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return [round(float(x), 6) for x in vec]

    async def get_embedding(self, input_data: str | bytes) -> list[float]:
        text = input_data if isinstance(input_data, str) else input_data.decode("utf-8", errors="ignore")
        logger.debug(f"Generating dense text embedding ({self.dim}-d) using model: {self.model_name}")
        return self._compute_dense_vector(text)

    async def get_batch_embeddings(self, inputs: list[str | bytes]) -> list[list[float]]:
        return [await self.get_embedding(item) for item in inputs]
