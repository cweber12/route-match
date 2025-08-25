import pytest
from httpx import AsyncClient
from main import app  # This is your FastAPI app instance

@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        res = await ac.get("/")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        res = await ac.get("/api/health")
        assert res.status_code == 200
        assert "status" in res.json()
