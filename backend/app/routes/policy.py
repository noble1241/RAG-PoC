from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from app.artifacts import dump_policy_json
from app.config import settings
from app.policy_extraction import enumerate_policies, extract_policy_per_service, scope_to_section
from app.policy_rag import ingest_policy
from app.schemas import (
    PolicyExtractionError,
    PolicyExtractionResponse,
    PolicyExtractionResult,
)
from app.uploads import BACKEND_DIR, read_and_convert_upload, resolve_backend_dir

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/documents/extract-policy",
    response_model=PolicyExtractionResponse,
    status_code=status.HTTP_200_OK,
)
async def extract_policy_endpoint(
    request: Request,
    file: Annotated[UploadFile, File()],
    sheet: str | None = Query(default=None, description="Restrict to the '## <sheet>' section"),
    policy: str | None = Query(default=None, description="Substring filter on enumerated policy names"),
) -> PolicyExtractionResponse:
    # Shared with the upload route: validates extension/size, converts to Markdown
    # (415/413/422 on bad input), and best-effort dumps the raw Markdown.
    filename, _ext, markdown, _converted_rel = await read_and_convert_upload(file)

    scoped = scope_to_section(markdown, sheet) if sheet else markdown
    model = settings.policy_extraction_model or settings.chat_model

    # Blocking OpenAI call -> run off the event loop.
    try:
        names = await asyncio.to_thread(enumerate_policies, scoped, model=model)
    except Exception as exc:
        logger.exception("Policy enumeration failed for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Policy enumeration failed: {exc}"
        )

    if policy:
        needle = policy.lower()
        names = [n for n in names if needle in n.lower()]
    if not names:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="No policy content detected"
        )

    out_dir = resolve_backend_dir(settings.policy_output_dir)
    results: list[PolicyExtractionResult] = []
    errors: list[PolicyExtractionError] = []

    for name in names:
        # Extraction is the failure-prone step; a per-policy failure must not sink the rest.
        try:
            doc = await asyncio.to_thread(
                extract_policy_per_service,
                scoped,
                model=model,
                extra_instructions=(
                    f"Extract ONLY the '{name}' policy/column; ignore all other policies/columns."
                ),
            )
        except Exception as exc:
            logger.exception("Extraction failed for policy %r in %s", name, filename)
            errors.append(PolicyExtractionError(policy_name=name, error=str(exc)))
            continue

        # JSON artifact is a convenience deliverable — non-fatal if the write fails.
        json_path: str | None = None
        try:
            saved = dump_policy_json(doc.model_dump_json(indent=2), filename, name, out_dir)
            json_path = (
                str(saved.relative_to(BACKEND_DIR))
                if saved.is_relative_to(BACKEND_DIR)
                else str(saved)
            )
        except Exception:
            logger.exception("Failed to write policy JSON for %r", name)

        # RAG ingest — a failure here is a per-policy error, not a whole-request failure.
        try:
            ingest = await ingest_policy(doc, source=f"{filename} [{name}]")
        except Exception as exc:
            logger.exception("Ingest failed for policy %r in %s", name, filename)
            errors.append(PolicyExtractionError(policy_name=name, error=f"ingest failed: {exc}"))
            continue

        results.append(
            PolicyExtractionResult(
                policy_name=name,
                chunk_count=ingest["chunk_count"],
                item_count=ingest["item_count"],
                json_path=json_path,
            )
        )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"All policy extractions failed: {[e.error for e in errors]}",
        )

    logger.info("Extracted %d policies from %s (%d errors)", len(results), filename, len(errors))
    return PolicyExtractionResponse(source=filename, policies=results, errors=errors)
