"""Integration tests for Static Frontend and Graph Visualization routes."""

from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config.settings import Settings

app = create_app(Settings(APP_ENV="test", DEBUG=True))
client = TestClient(app)


def test_root_serves_frontend_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Multimodal Graph RAG" in response.text
    assert "vis-network" in response.text


def test_static_css_served():
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    assert "--primary-indigo" in response.text


def test_static_js_served():
    response = client.get("/static/js/app.js")
    assert response.status_code == 200
    assert "application/javascript" in response.headers["content-type"] or "text/javascript" in response.headers["content-type"]


def test_graph_overview_endpoint():
    response = client.get("/api/v1/graph/overview")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert "total_nodes" in data
    assert data["total_nodes"] > 0
    assert len(data["nodes"]) > 0
    # Check node schema matches Vis-Network requirements
    assert "id" in data["nodes"][0]
    assert "label" in data["nodes"][0]
    assert "group" in data["nodes"][0]
