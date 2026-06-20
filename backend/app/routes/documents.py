from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.chunking import chunk_text
from app.config import settings
from app.llm import embed_texts
from app.schemas import IngestResponse, IngestTextRequest
from app.vectorstore import upsert_chunks

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    import io
    import re

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout") or ""
        # Collapse runs of multiple spaces into one (PDF spacing artifacts)
        text = re.sub(r" {2,}", " ", text)
        # Remove spaces between single characters (e.g. "h e l l o" → "hello")
        text = re.sub(r"(?<!\w)(\w) (?=\w )", r"\1", text)
        text = re.sub(r"(?<!\w)(\w) (?=\w\b)", r"\1", text)
        pages.append(text.strip())
    return "\n\n".join(p for p in pages if p)


async def _ingest_text(text: str, source: str) -> IngestResponse:
    chunks = chunk_text(
        text=text,
        source=source,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="No text content found")

    doc_id = hashlib.sha256(source.encode()).hexdigest()[:16]
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    texts = [c.text for c in chunks]
    embeddings = await embed_texts(texts)

    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "source": c.source,
            "chunk_index": c.chunk_index,
            "ingested_at": ts,
            "content_hash": c.content_hash,
        }
        for c in chunks
    ]

    await upsert_chunks(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    total_tokens = sum(c.token_count for c in chunks)
    logger.info("Ingested %s: %d chunks, %d tokens", source, len(chunks), total_tokens)

    return IngestResponse(
        document_id=doc_id,
        source=source,
        chunk_count=len(chunks),
        tokens_processed=total_tokens,
    )


@router.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(request: Request, body: IngestTextRequest) -> IngestResponse:
    return await _ingest_text(text=body.text, source=body.source)


@router.post("/documents/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(
    request: Request,
    file: Annotated[UploadFile, File()],
) -> IngestResponse:
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in {"txt", "md", "pdf"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .txt, .md, and .pdf files are supported",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large (max 10 MB)")

    try:
        if ext == "pdf":
            text = _extract_pdf_text(data)
        else:
            text = data.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.exception("Failed to parse file %s", filename)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"Could not parse file: {exc}")

    return await _ingest_text(text=text, source=filename)
