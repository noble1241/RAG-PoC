"""Unit tests for the shared upload validate+convert helper."""
from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from app.uploads import read_and_convert_upload


async def test_read_and_convert_returns_markdown_tuple(tmp_path, monkeypatch):
    import app.uploads as up

    monkeypatch.setattr(up.settings, "converted_output_dir", str(tmp_path))
    f = UploadFile(file=BytesIO(b"name,role\nAlice,eng\n"), filename="people.csv")
    filename, ext, markdown, converted_rel = await read_and_convert_upload(f)
    assert filename == "people.csv"
    assert ext == "csv"
    assert "Alice" in markdown


async def test_read_and_convert_rejects_unsupported_extension():
    f = UploadFile(file=BytesIO(b"MZ"), filename="bad.exe")
    with pytest.raises(HTTPException) as exc:
        await read_and_convert_upload(f)
    assert exc.value.status_code == 415


async def test_read_and_convert_sanitizes_conversion_error_detail():
    # %PDF magic bytes commit MarkItDown to the PDF converter (no plaintext
    # fallback); the malformed body then makes pdfminer raise internally.
    f = UploadFile(file=BytesIO(b"%PDF-1.7 broken\x00\x01\x02"), filename="broken.pdf")
    with pytest.raises(HTTPException) as exc:
        await read_and_convert_upload(f)
    assert exc.value.status_code == 422
    assert exc.value.detail == (
        "Could not parse file: the document is invalid or unsupported by the converter."
    )
