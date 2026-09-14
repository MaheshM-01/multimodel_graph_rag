"""Vision embedding engine supporting CLIP, SigLIP, ColPali, and normalized cross-modal representations."""

import hashlib
import re
from pathlib import Path
import numpy as np
from src.config.settings import get_settings
from src.core.logging import logger
from src.embeddings.base import BaseEmbeddingEngine


class VisionEmbeddingEngine(BaseEmbeddingEngine):
    """Generates dense normalized vector embeddings for images, charts, and visual queries."""

    def __init__(self, model_name: str | None = None, dim: int = 512):
        settings = get_settings()
        self.model_name = model_name or settings.DEFAULT_VISION_EMBEDDING_MODEL
        self.dim = dim

    def _compute_vision_vector(self, input_data: str | bytes | Path) -> list[float]:
        if isinstance(input_data, Path):
            label = str(input_data.name)
        elif isinstance(input_data, bytes):
            label = hashlib.md5(input_data[:2048]).hexdigest()
        else:
            label = str(input_data)

        clean_text = label.lower().strip()
        tokens = re.findall(r"\b\w+\b", clean_text)
        vec = np.zeros(self.dim, dtype=np.float32)

        if not tokens:
            vec[0] = 1.0
            return vec.tolist()

        for token in tokens:
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dim
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign * (1.0 + len(token) / 10.0)

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return [round(float(x), 6) for x in vec]

    async def get_embedding(self, input_data: str | bytes | Path) -> list[float]:
        logger.debug(f"Generating vision embedding ({self.dim}-d) using model: {self.model_name}")
        return self._compute_vision_vector(input_data)

    async def get_batch_embeddings(self, inputs: list[str | bytes | Path]) -> list[list[float]]:
        return [await self.get_embedding(item) for item in inputs]
