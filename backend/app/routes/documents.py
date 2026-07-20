from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.artifacts import dump_markdown
from app.config import settings
from app.conversion import SUPPORTED_EXTENSIONS, get_converter
from app.llm import embed_texts
from app.manifest import record_ingestion
from app.schemas import IngestResponse, IngestTextRequest
from app.structuring import structure_document
from app.vectorstore import upsert_chunks

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# backend/ root — resolves a relative converted_output_dir predictably,
# regardless of the process CWD. (routes/ -> app/ -> backend/)
_BACKEND_DIR = Path(__file__).resolve().parents[2]


def _resolve_converted_dir() -> Path:
    out_dir = Path(settings.converted_output_dir)
    if not out_dir.is_absolute():
        out_dir = _BACKEND_DIR / out_dir
    return out_dir


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
            _resolve_converted_dir() / settings.manifest_filename,
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
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File too large (max 10 MB)",
        )

    try:
        markdown = await get_converter().to_markdown(data, filename)
    except asyncio.TimeoutError:
        logger.warning("Conversion timed out for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="File conversion timed out",
        )
    except Exception as exc:
        logger.exception("Failed to convert file %s", filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Could not parse file: {exc}",
        )

    # Dump the raw converted Markdown to disk for inspection (best-effort — a
    # failure here must never break ingestion).
    converted_rel: str | None = None
    if settings.save_converted_markdown:
        try:
            saved = dump_markdown(markdown, filename, _resolve_converted_dir())
            converted_rel = (
                str(saved.relative_to(_BACKEND_DIR))
                if saved.is_relative_to(_BACKEND_DIR)
                else str(saved)
            )
            logger.info("Saved converted markdown to %s", saved)
        except Exception:
            logger.exception("Failed to save converted markdown for %s", filename)

    return await _ingest_text(
        text=markdown, source=filename, file_type=ext, converted_markdown=converted_rel
    )
