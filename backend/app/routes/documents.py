from __future__ import annotations

import datetime
import hashlib
import json
import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.config import settings
from app.llm import embed_texts
from app.manifest import record_ingestion
from app.schemas import IngestResponse, IngestTextRequest
from app.structuring import structure_document
from app.uploads import read_and_convert_upload, resolve_backend_dir
from app.vectorstore import upsert_chunks

logger = logging.getLogger(__name__)
router = APIRouter()


async def _ingest_text(
    text: str,
    source: str,
    file_type: str,
    converted_markdown: str | None = None,
) -> IngestResponse:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    doc = structure_document(
        markdown=text,
        source=source,
        file_type=file_type,
        ingested_at=ts,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not doc.narrative_chunks and not doc.table_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No text content found",
        )

    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for c in doc.narrative_chunks:
        texts.append(c.text)
        ids.append(c.chunk_id)
        metadatas.append(
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "ingested_at": ts,
                "content_hash": c.content_hash,
                "heading_path": " > ".join(c.heading_path),
                "content_type": "narrative",
            }
        )
    for c in doc.table_chunks:
        texts.append(c.text)
        ids.append(c.chunk_id)
        metadatas.append(
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "ingested_at": ts,
                "content_hash": c.content_hash,
                "heading_path": " > ".join(c.heading_path),
                "content_type": "table-summary",
                "table_rows": json.dumps(c.rows),
            }
        )

    embeddings = await embed_texts(texts)
    await upsert_chunks(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    # Manifest update (best-effort — a failure here must never break ingestion).
    try:
        record_ingestion(
            resolve_backend_dir(settings.converted_output_dir) / settings.manifest_filename,
            source,
            {
                "file_type": file_type,
                "ingested_at": ts,
                "converted_markdown": converted_markdown,
                "chunk_ids": [c.chunk_id for c in doc.narrative_chunks],
                "table_ids": [c.chunk_id for c in doc.table_chunks],
            },
        )
    except Exception:
        logger.exception("Failed to update manifest for %s", source)

    total_tokens = sum(c.token_count for c in doc.narrative_chunks) + sum(
        c.token_count for c in doc.table_chunks
    )
    doc_id = hashlib.sha256(source.encode()).hexdigest()[:16]
    logger.info(
        "Ingested %s: %d chunks (%d narrative, %d table), %d tokens",
        source,
        len(texts),
        len(doc.narrative_chunks),
        len(doc.table_chunks),
        total_tokens,
    )
    return IngestResponse(
        document_id=doc_id,
        source=source,
        chunk_count=len(texts),
        tokens_processed=total_tokens,
    )


@router.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(request: Request, body: IngestTextRequest) -> IngestResponse:
    return await _ingest_text(text=body.text, source=body.source, file_type="text")


@router.post("/documents/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(
    request: Request,
    file: Annotated[UploadFile, File()],
) -> IngestResponse:
    filename, ext, markdown, converted_rel = await read_and_convert_upload(file)
    return await _ingest_text(
        text=markdown, source=filename, file_type=ext, converted_markdown=converted_rel
    )
