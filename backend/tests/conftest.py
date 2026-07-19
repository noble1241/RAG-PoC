import os

import pytest
from httpx import AsyncClient, ASGITransport

# Set env before importing app modules
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8000")


@pytest.fixture(autouse=True)
def _isolate_converted_output(tmp_path_factory, monkeypatch):
    """Point converted-markdown dumps at a temp dir so tests never write into
    the real backend/converted_output/. Tests that assert on the dump can
    still override converted_output_dir with their own tmp_path."""
    from app.config import settings

    d = tmp_path_factory.mktemp("converted_output")
    monkeypatch.setattr(settings, "converted_output_dir", str(d))


@pytest.fixture()
async def client():
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
