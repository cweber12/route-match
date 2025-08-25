import pytest
from fastapi.testclient import TestClient
from main import app  # This is your FastAPI app instance

def test_root():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}

def test_health_check():
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        assert "status" in res.json()
