from __future__ import annotations

import asyncio
import io
import os
from typing import Protocol

from markitdown import MarkItDown

from app.config import settings

# Single source of truth for accepted upload types (no dot, lowercase).
# Must stay in sync with the markitdown extras installed in requirements.txt.
SUPPORTED_EXTENSIONS: set[str] = {
    "txt", "md", "csv", "pdf", "docx", "xlsx", "xls", "pptx",
}


class DocumentConverter(Protocol):
    async def to_markdown(self, data: bytes, filename: str) -> str: ...


class LocalMarkItDownConverter:
    """In-process MarkItDown converter. Blocking conversion is offloaded to a
    worker thread so it never freezes the async event loop."""

    def __init__(self, timeout_seconds: int) -> None:
        self._md = MarkItDown()
        self._timeout = timeout_seconds

    async def to_markdown(self, data: bytes, filename: str) -> str:
        loop = asyncio.get_running_loop()
        ext = os.path.splitext(filename)[1]  # e.g. ".pdf" — a hint for detection

        def _convert() -> str:
            result = self._md.convert_stream(io.BytesIO(data), file_extension=ext)
            return result.text_content or ""

        return await asyncio.wait_for(
            loop.run_in_executor(None, _convert),
            timeout=self._timeout,
        )


_default_converter: LocalMarkItDownConverter | None = None


def get_converter() -> DocumentConverter:
    global _default_converter
    if _default_converter is None:
        _default_converter = LocalMarkItDownConverter(
            settings.conversion_timeout_seconds
        )
    return _default_converter
