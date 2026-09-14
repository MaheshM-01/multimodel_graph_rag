"""Integration tests for FastAPI endpoints."""

import io
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config.settings import Settings

app = create_app(Settings(APP_ENV="test", DEBUG=True))
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data


def test_readiness_endpoint():
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True


def test_rag_query_endpoint():
    payload = {
        "query": "What is the revenue growth for Cloud Services?",
        "temperature": 0.1,
    }
    response = client.post("/api/v1/rag/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "grounding" in data


def test_async_ingest_and_job_status():
    # 1. Upload mock file to /api/v1/ingest/upload
    fake_file = io.BytesIO(b"%PDF-1.4 mock pdf content")
    response = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("test_report.pdf", fake_file, "application/pdf")},
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    job_id = data["job_id"]
    assert data["status"] == "pending"

    # 2. Poll /api/v1/jobs/{job_id}
    job_response = client.get(f"/api/v1/jobs/{job_id}")
    assert job_response.status_code == 200
    job_data = job_response.json()
    assert job_data["job_id"] == job_id
    assert job_data["filename"] == "test_report.pdf"


def test_documents_management_endpoints():
    # 1. List documents
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    docs = response.json()
    assert isinstance(docs, list)

    # 2. Write a temporary doc to verify view and delete
    from pathlib import Path
    test_path = Path("./data/media_store/temp_kb_test.txt")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_bytes(b"Hello Knowledge Base Test")

    # 3. View document
    view_res = client.get("/api/v1/documents/temp_kb_test.txt/view")
    assert view_res.status_code == 200
    assert b"Hello Knowledge Base Test" in view_res.content

    # 4. Delete document
    del_res = client.delete("/api/v1/documents/temp_kb_test.txt")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 5. Confirm 404 after deletion
    view_after_del = client.get("/api/v1/documents/temp_kb_test.txt/view")
    assert view_after_del.status_code == 404
