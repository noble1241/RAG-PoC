import os

import pytest
from httpx import AsyncClient, ASGITransport

# Set env before importing app modules
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8000")


@pytest.fixture()
async def client():
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
