from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.artifacts import dump_markdown
from app.config import settings
from app.conversion import SUPPORTED_EXTENSIONS, get_converter

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
# app/ -> backend/ ; resolves a relative output dir regardless of process CWD.
BACKEND_DIR = Path(__file__).resolve().parents[1]


def resolve_backend_dir(name: str) -> Path:
    """Resolve a possibly-relative directory name against backend/."""
    d = Path(name)
    return d if d.is_absolute() else BACKEND_DIR / d


async def read_and_convert_upload(file: UploadFile) -> tuple[str, str, str, str | None]:
    """Validate, read, and convert an uploaded file to Markdown.

    Returns ``(filename, ext, markdown, converted_rel)`` where ``converted_rel`` is the
    backend-relative path of the best-effort Markdown dump (or ``None`` if the dump is
    disabled or failed). Raises ``HTTPException`` 415 (unsupported extension), 413 (over
    10 MB), or 422 (conversion error/timeout). Shared by the upload and policy routes.
    """
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="File too large (max 10 MB)"
        )

    try:
        markdown = await get_converter().to_markdown(data, filename)
    except asyncio.TimeoutError:
        logger.warning("Conversion timed out for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="File conversion timed out"
        )
    except Exception:
        logger.exception("Failed to convert file %s", filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Could not parse file: the document is invalid or unsupported by the converter.",
        )

    converted_rel: str | None = None
    if settings.save_converted_markdown:
        try:
            saved = dump_markdown(markdown, filename, resolve_backend_dir(settings.converted_output_dir))
            converted_rel = (
                str(saved.relative_to(BACKEND_DIR)) if saved.is_relative_to(BACKEND_DIR) else str(saved)
            )
            logger.info("Saved converted markdown to %s", saved)
        except Exception:
            logger.exception("Failed to save converted markdown for %s", filename)

    return filename, ext, markdown, converted_rel
