import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

import hashlib
import pytest

from app.chunking import chunk_text, _make_id


def test_chunk_sizes_respected():
    text = "word " * 1000
    chunks = chunk_text(text, source="test.txt", chunk_size=100, chunk_overlap=10)
    for c in chunks[:-1]:  # last chunk may be smaller
        assert c.token_count == 100


def test_overlap_correct():
    text = "word " * 500
    chunks = chunk_text(text, source="test.txt", chunk_size=100, chunk_overlap=20)
    # Second chunk starts 80 tokens after first chunk (100 - 20)
    assert len(chunks) > 1
    total_tokens = sum(c.token_count for c in chunks)
    # With overlap, total > source tokens
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    source_tokens = len(enc.encode(text))
    assert total_tokens > source_tokens or len(chunks) == 1


def test_deterministic_ids():
    text = "Some sample text for testing " * 50
    chunks1 = chunk_text(text, source="doc.txt", chunk_size=50, chunk_overlap=5)
    chunks2 = chunk_text(text, source="doc.txt", chunk_size=50, chunk_overlap=5)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_different_sources_different_ids():
    text = "Same content " * 50
    chunks_a = chunk_text(text, source="a.txt", chunk_size=50, chunk_overlap=5)
    chunks_b = chunk_text(text, source="b.txt", chunk_size=50, chunk_overlap=5)
    ids_a = {c.chunk_id for c in chunks_a}
    ids_b = {c.chunk_id for c in chunks_b}
    assert ids_a.isdisjoint(ids_b)


def test_empty_text_returns_no_chunks():
    chunks = chunk_text("", source="empty.txt", chunk_size=100, chunk_overlap=10)
    assert chunks == []


def test_short_text_single_chunk():
    chunks = chunk_text("Hello world", source="short.txt", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
