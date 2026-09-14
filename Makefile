.PHONY: help install dev test lint format clean docker-up docker-down

help:
	@echo "Multimodal Graph RAG Development Commands:"
	@echo "  make install     - Install all project and development dependencies"
	@echo "  make dev         - Run local FastAPI development server with hot-reload"
	@echo "  make test        - Run test suite with pytest"
	@echo "  make lint        - Check code formatting and static analysis with Ruff"
	@echo "  make format      - Auto-format code with Ruff"
	@echo "  make docker-up   - Start local services (Neo4j, Qdrant, MinIO) in background"
	@echo "  make docker-down - Stop all running docker services"
	@echo "  make clean       - Remove cached files and build artifacts"

install:
	pip install -e ".[dev]"

dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

docker-up:
	docker-compose up -d neo4j qdrant minio

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
