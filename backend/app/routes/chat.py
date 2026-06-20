from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.config import settings
from app.llm import embed_texts, stream_answer
from app.schemas import ChatRequest, SourceChunk
from app.vectorstore import collection_count, query_collection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    count = await collection_count()
    if count == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No documents ingested yet. Please add documents before querying.",
        )

    query_embedding = (await embed_texts([body.query]))[0]
    results = await query_collection(
        query_embedding=query_embedding,
        n_results=min(settings.top_k, count),
    )

    # Unpack chromadb query result (batched: index 0 = our single query)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # chromadb cosine distance → similarity score (1 - distance)
    chunks: list[SourceChunk] = [
        SourceChunk(
            source=meta.get("source", "unknown"),
            chunk_index=int(meta.get("chunk_index", 0)),
            content=doc,
            score=round(1.0 - dist, 4),
        )
        for doc, meta, dist in zip(docs, metas, distances)
    ]

    answer_stream = await stream_answer(query=body.query, chunks=chunks)

    async def _sse_generator():
        # First send sources as a JSON event
        sources_payload = json.dumps({"type": "sources", "sources": [c.model_dump() for c in chunks]})
        yield f"data: {sources_payload}\n\n"

        async for token in answer_stream:
            payload = json.dumps({"type": "token", "content": token})
            yield f"data: {payload}\n\n"

        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
