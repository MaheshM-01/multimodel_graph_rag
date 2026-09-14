"""Cross-modal embedding fusion and projection."""

from src.core.logging import logger


class MultimodalFusionEngine:
    """Projects text and vision embeddings into a shared joint-retrieval representation."""

    def __init__(self):
        logger.info("Initializing MultimodalFusionEngine")

    def fuse_embeddings(self, text_vec: list[float], vision_vec: list[float], alpha: float = 0.5) -> list[float]:
        """Fuses text and visual vectors with weighting parameter alpha."""
        if len(text_vec) != len(vision_vec):
            # When dimensionality differs, return concatenated or projected representation
            return text_vec + vision_vec
        return [alpha * t + (1 - alpha) * v for t, v in zip(text_vec, vision_vec, strict=False)]
