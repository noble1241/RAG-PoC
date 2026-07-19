import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

import asyncio

from app.conversion import (
    SUPPORTED_EXTENSIONS,
    LocalMarkItDownConverter,
    get_converter,
)


def test_supported_extensions_are_policy_formats():
    assert SUPPORTED_EXTENSIONS == {
        "txt", "md", "csv", "pdf", "docx", "xlsx", "xls", "pptx",
    }


def test_csv_converts_to_markdown_table():
    conv = LocalMarkItDownConverter(timeout_seconds=30)
    data = b"name,role\nAlice,engineer\nBob,manager\n"
    md = asyncio.run(conv.to_markdown(data, "people.csv"))
    assert "Alice" in md
    assert "|" in md  # markdown table pipes


def test_plaintext_passthrough():
    conv = LocalMarkItDownConverter(timeout_seconds=30)
    md = asyncio.run(conv.to_markdown(b"hello world", "note.txt"))
    assert "hello world" in md


def test_get_converter_is_singleton():
    assert get_converter() is get_converter()
