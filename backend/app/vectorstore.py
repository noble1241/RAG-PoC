from __future__ import annotations

import asyncio
import logging
from typing import Any

import chromadb

from app.config import settings

logger = logging.getLogger(__name__)

# chromadb 1.x: synchronous HttpClient; we run blocking calls in thread pool
_client: chromadb.HttpClient | None = None
_collection: Any | None = None


def _get_client() -> chromadb.HttpClient:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    return _client


def _get_collection() -> Any:
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


async def ping_chroma() -> bool:
    """Return True if Chroma is reachable."""
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, lambda: _get_client().heartbeat())
        return True
    except Exception as exc:
        logger.warning("Chroma ping failed: %s", exc)
        return False


async def upsert_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    loop = asyncio.get_running_loop()

    def _upsert() -> None:
        col = _get_collection()
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    await loop.run_in_executor(None, _upsert)


async def query_collection(
    query_embedding: list[float],
    n_results: int,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()

    def _query() -> dict[str, Any]:
        col = _get_collection()
        return col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    return await loop.run_in_executor(None, _query)


async def collection_count() -> int:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _get_collection().count())
