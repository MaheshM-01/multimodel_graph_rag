"""Structured logging configuration."""

import logging
import sys

try:
    from loguru import logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    logger = logging.getLogger("multimodal_graph_rag")  # type: ignore


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    if LOGURU_AVAILABLE:
        logger.remove()
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level.upper(),
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )


__all__ = ["logger", "setup_logging"]
