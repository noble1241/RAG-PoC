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


def test_indented_code_block_content_is_preserved():
    md = "# Doc\n\nIntro paragraph.\n\n    indented_code_line_alpha = 1\n    indented_code_line_beta = 2\n\nAfter code.\n"
    chunks = chunk_markdown(md, source="c.md", chunk_size=500, chunk_overlap=10)
    joined = "\n".join(c.text for c in chunks)
    assert "indented_code_line_alpha" in joined
    assert "indented_code_line_beta" in joined


import tiktoken


def _tok(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def test_oversized_section_splits_with_overlap():
    body = " ".join(f"word{i}" for i in range(2000))
    md = f"# Big\n\n{body}\n"
    chunks = chunk_markdown(md, source="big.md", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 200 + 5  # small tolerance for prefix/decoding
    # overlap: total tokens across chunks exceeds a single pass of the source
    assert sum(c.token_count for c in chunks) > _tok(body)


def test_multi_block_section_respects_cap():
    # budget = chunk_size(200) - prefix_tokens("Sec\n\n") = 198; overlap = 20.
    # p1 ~150 tokens (< budget). p2 ~186 tokens: < budget on its own, but
    # > budget - overlap (178), so carrying the overlap tail + p2 would
    # exceed budget and must be caught by the packing-loop re-check.
    p1 = " ".join(f"alpha{i}" for i in range(75))
    p2 = " ".join(f"beta{i}" for i in range(93))
    assert _tok(p1) < 198
    assert 178 < _tok(p2) <= 198
    md = f"# Sec\n\n{p1}\n\n{p2}\n"
    chunks = chunk_markdown(md, source="m.md", chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 205, f"chunk {c.chunk_index} has {c.token_count} tokens"


def test_hash_inside_code_fence_is_not_a_heading():
    md = "# Real\n\n```python\n# this is a comment, not a heading\nx = 1\n```\n"
    chunks = chunk_markdown(md, source="code.md", chunk_size=500, chunk_overlap=10)
    # Only one heading path exists; the '#' comment must not create a section.
    paths = {tuple(c.heading_path) for c in chunks}
    assert paths == {("Real",)}


def test_large_table_splits_by_rows_repeating_header():
    rows = "\n".join(f"| r{i} | v{i} |" for i in range(200))
    md = f"# T\n\n| name | value |\n| --- | --- |\n{rows}\n"
    chunks = chunk_markdown(md, source="t.md", chunk_size=120, chunk_overlap=0)
    table_chunks = [c for c in chunks if "| name | value |" in c.text]
    assert len(table_chunks) > 1  # split into multiple parts
    for c in table_chunks:
        # every part re-emits the header + separator so it stays a valid table
        assert "| name | value |" in c.text
        assert "| --- | --- |" in c.text
