"""Observability and distributed tracing for multi-hop graph, vector, and LLM calls."""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
import functools
import time
from typing import Any
from src.config.settings import get_settings
from src.core.logging import logger


class TelemetryManager:
    """Manages spans, execution metrics, and tracing metadata across pipeline steps."""

    def __init__(self):
        settings = get_settings()
        self.enabled = settings.TELEMETRY_ENABLED
        self.service_name = settings.OTEL_SERVICE_NAME

    @asynccontextmanager
    async def trace_span(self, span_name: str, attributes: dict[str, Any] | None = None) -> AsyncGenerator[None, None]:
        """Context manager creating a trace span for tracking latency and attributes."""
        attrs = attributes or {}
        start_time = time.perf_counter()
        if self.enabled:
            logger.debug(f"[Trace:Start] {span_name} | {attrs}")
        try:
            yield
        finally:
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            if self.enabled:
                logger.debug(f"[Trace:End] {span_name} completed in {elapsed}ms")


telemetry = TelemetryManager()


def trace_action(name: str | None = None):
    """Decorator to trace async functions and track latency."""

    def decorator(func: Callable):
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with telemetry.trace_span(span_name, {"args": len(args)}):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
