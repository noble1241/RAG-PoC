"""API-level tests with OpenAI and Chroma mocked."""
from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ready_chroma_up(client):
    with patch("app.routes.health.ping_chroma", return_value=True):
        r = await client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["chroma"] == "ok"


@pytest.mark.asyncio
async def test_ready_chroma_down(client):
    with patch("app.routes.health.ping_chroma", return_value=False):
        r = await client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "degraded"
    assert data["chroma"] == "unreachable"


@pytest.mark.asyncio
async def test_ingest_text_validation_empty(client):
    r = await client.post("/documents", json={"text": "", "source": "test"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_ingest_text_success(client):
    fake_embedding = [0.1] * 1536

    with (
        patch("app.routes.documents.embed_texts", new_callable=AsyncMock, return_value=[fake_embedding] * 10),
        patch("app.routes.documents.upsert_chunks", new_callable=AsyncMock),
    ):
        r = await client.post(
            "/documents",
            json={"text": "This is sample content. " * 50, "source": "test.txt"},
        )
    assert r.status_code == 201
    data = r.json()
    assert data["source"] == "test.txt"
    assert data["chunk_count"] >= 1
    assert data["tokens_processed"] > 0
    assert len(data["document_id"]) == 16


@pytest.mark.asyncio
async def test_chat_empty_collection(client):
    with patch("app.routes.chat.collection_count", new_callable=AsyncMock, return_value=0):
        r = await client.post("/chat", json={"query": "What is X?"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_chat_query_too_long(client):
    r = await client.post("/chat", json={"query": "x" * 5000})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_chat_streams_sse(client):
    fake_embedding = [0.1] * 1536
    fake_results = {
        "documents": [["Chunk content about X."]],
        "metadatas": [[{"source": "doc.txt", "chunk_index": 0}]],
        "distances": [[0.1]],
    }

    async def fake_stream():
        for token in ["Hello", " world", "!"]:
            yield token

    with (
        patch("app.routes.chat.collection_count", new_callable=AsyncMock, return_value=5),
        patch("app.routes.chat.embed_texts", new_callable=AsyncMock, return_value=[fake_embedding]),
        patch("app.routes.chat.query_collection", new_callable=AsyncMock, return_value=fake_results),
        patch("app.routes.chat.stream_answer", new_callable=AsyncMock, return_value=fake_stream()),
    ):
        r = await client.post("/chat", json={"query": "What is X?"})

    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    lines = r.text.strip().split("\n")
    data_lines = [l for l in lines if l.startswith("data: ")]
    events = [json.loads(l[6:]) for l in data_lines]

    types = [e["type"] for e in events]
    assert "sources" in types
    assert "token" in types
    assert "done" in types

    # sources event contains chunks
    sources_event = next(e for e in events if e["type"] == "sources")
    assert len(sources_event["sources"]) == 1
    assert sources_event["sources"][0]["source"] == "doc.txt"


@pytest.mark.asyncio
async def test_ingest_unsupported_file_type(client):
    from io import BytesIO
    r = await client.post(
        "/documents/upload",
        files={"file": ("test.exe", BytesIO(b"data"), "application/octet-stream")},
    )
    assert r.status_code == 415
