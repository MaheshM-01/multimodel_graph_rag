"""Pytest test configuration and shared fixtures."""

import pytest
from src.config.settings import Settings
from src.domain.multimodal import Document
from src.core.constants import ModalityType


@pytest.fixture
def test_settings() -> Settings:
    """Test configuration settings."""
    return Settings(
        APP_ENV="test",
        DEBUG=True,
        NEO4J_URI="bolt://localhost:7687",
        QDRANT_HOST="localhost",
    )


@pytest.fixture
def sample_document() -> Document:
    """Sample multimodal document fixture."""
    return Document(
        title="Test Document",
        source_uri="test://sample.pdf",
        modality=ModalityType.TEXT,
        mime_type="application/pdf",
    )
