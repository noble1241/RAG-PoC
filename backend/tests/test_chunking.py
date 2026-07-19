import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

from app.chunking import Chunk, chunk_markdown


def test_empty_text_returns_no_chunks():
    assert chunk_markdown("", source="empty.md", chunk_size=100, chunk_overlap=10) == []


def test_single_paragraph_single_chunk():
    chunks = chunk_markdown("Hello world.", source="s.md", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert "Hello world." in chunks[0].text


def test_heading_breadcrumb_prefixed_into_text():
    md = "# Leave Policy\n\n## Parental Leave\n\nGive eight weeks notice.\n"
    chunks = chunk_markdown(md, source="p.md", chunk_size=200, chunk_overlap=10)
    body = next(c for c in chunks if "eight weeks" in c.text)
    assert body.heading_path == ["Leave Policy", "Parental Leave"]
    assert body.text.startswith("Leave Policy > Parental Leave\n\n")


def test_sibling_heading_resets_path():
    md = (
        "# Policy\n\n## Leave\n\nLeave body text here.\n\n"
        "## Pay\n\nPay body text here.\n"
    )
    chunks = chunk_markdown(md, source="p.md", chunk_size=200, chunk_overlap=10)
    leave = next(c for c in chunks if "Leave body" in c.text)
    pay = next(c for c in chunks if "Pay body" in c.text)
    assert leave.heading_path == ["Policy", "Leave"]
    assert pay.heading_path == ["Policy", "Pay"]


def test_deterministic_ids():
    md = "# A\n\nSome sample body text for testing determinism.\n"
    a = chunk_markdown(md, source="doc.md", chunk_size=200, chunk_overlap=10)
    b = chunk_markdown(md, source="doc.md", chunk_size=200, chunk_overlap=10)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_different_sources_different_ids():
    md = "# A\n\nSame content body here for the test.\n"
    a = chunk_markdown(md, source="a.md", chunk_size=200, chunk_overlap=10)
    b = chunk_markdown(md, source="b.md", chunk_size=200, chunk_overlap=10)
    assert {c.chunk_id for c in a}.isdisjoint({c.chunk_id for c in b})
