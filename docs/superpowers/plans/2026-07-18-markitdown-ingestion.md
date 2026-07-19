# MarkItDown Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pypdf-based, blind-token-window ingestion with a MarkItDown converter (behind an isolation seam) plus a Markdown-structure-aware chunker that carries heading breadcrumbs.

**Architecture:** Uploaded files are converted to structured Markdown by MarkItDown running in a thread pool behind a `DocumentConverter` Protocol. A rewritten `chunk_markdown()` splits that Markdown on heading hierarchy, keeps tables/code intact, and prefixes each chunk with its heading breadcrumb before embedding. Embed (`llm.py`) and upsert (`vectorstore.py`) are unchanged.

**Tech Stack:** FastAPI, MarkItDown (`markitdown[docx,pptx,xlsx,xls,pdf]`), `markdown-it-py`, tiktoken (`cl100k_base`), ChromaDB, pytest + pytest-asyncio.

**Design spec:** `docs/superpowers/specs/2026-07-18-markitdown-ingestion-design.md`

## Global Constraints

- Supported upload extensions (single source of truth): `txt, md, csv, pdf, docx, xlsx, xls, pptx`.
- Max upload size: `10 MB` (`MAX_UPLOAD_BYTES = 10 * 1024 * 1024`), unchanged.
- Tokenizer: tiktoken `cl100k_base`, unchanged.
- Deterministic chunk IDs: `chunk_id = sha256("{source}::{index}::{content_hash}")[:32]`, `content_hash = sha256(chunk_text)[:16]` — must stay stable across runs so re-ingest overwrites.
- Breadcrumb is prefixed INTO the embedded/stored chunk text as `"<h1> > <h2> > ...\n\n<body>"`.
- Conversion runs off the event loop (`run_in_executor`) with an `asyncio.wait_for` timeout.
- All new tests must run fully offline (no OpenAI/Chroma/network). Mock `embed_texts`/`upsert_chunks` in API tests.
- Working branch: `feat/markitdown-ingestion`. All commands assume CWD `backend/` unless a path says otherwise.

---

### Task 1: Dependencies and config setting

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/config.py:26-30` (RAG tuning block)

**Interfaces:**
- Produces: `settings.conversion_timeout_seconds: int` (default `30`), consumed by Task 2.

- [ ] **Step 1: Edit `backend/requirements.txt`**

Remove the `pypdf==6.13.3` line. Add these two lines (unpinned MarkItDown extras + parser):

```
markitdown[docx,pptx,xlsx,xls,pdf]
markdown-it-py
```

- [ ] **Step 2: Add the timeout setting to `config.py`**

In `Settings`, under the `# RAG tuning` block (after `chunk_overlap`), add:

```python
    # RAG tuning
    top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 50
    conversion_timeout_seconds: int = 30
```

- [ ] **Step 3: Install and verify import**

Run: `pip install -r requirements.txt`
Expected: installs MarkItDown, markdown-it-py, pandas, mammoth, pdfminer.six, pdfplumber, python-pptx, magika, etc. with no error.

Run: `python -c "from markitdown import MarkItDown; from markdown_it import MarkdownIt; print('ok')"`
Expected: prints `ok`.

Run: `python -c "from app.config import settings; print(settings.conversion_timeout_seconds)"` with `OPENAI_API_KEY=sk-test-key` in env.
Expected: prints `30`.

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt backend/app/config.py
git commit -m "chore: swap pypdf for markitdown deps, add conversion timeout setting"
```

---

### Task 2: DocumentConverter seam (`conversion.py`)

**Files:**
- Create: `backend/app/conversion.py`
- Create: `backend/tests/test_conversion.py`

**Interfaces:**
- Consumes: `settings.conversion_timeout_seconds` (Task 1).
- Produces:
  - `SUPPORTED_EXTENSIONS: set[str]` (no dots, lowercase).
  - `class DocumentConverter(Protocol)` with `async def to_markdown(self, data: bytes, filename: str) -> str`.
  - `class LocalMarkItDownConverter` implementing it, ctor `(timeout_seconds: int)`.
  - `def get_converter() -> DocumentConverter` — module-level singleton.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_conversion.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_conversion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.conversion'`.

- [ ] **Step 3: Write `backend/app/conversion.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_conversion.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/conversion.py backend/tests/test_conversion.py
git commit -m "feat: add MarkItDown converter behind DocumentConverter seam"
```

---

### Task 3: Markdown chunker — sections, breadcrumbs, IDs

**Files:**
- Modify: `backend/app/chunking.py` (full rewrite)
- Modify: `backend/tests/test_chunking.py` (full rewrite)

**Interfaces:**
- Produces:
  - `@dataclass Chunk` with fields `text, source, chunk_index, token_count, content_hash, chunk_id, heading_path: list[str]`.
  - `def chunk_markdown(text: str, source: str, chunk_size: int, chunk_overlap: int, encoding_name: str = "cl100k_base") -> list[Chunk]`.
  - `def _make_id(source: str, chunk_index: int, content_hash: str) -> str` (unchanged behavior).
- Note: this task installs a NAIVE `_section_bodies` that emits one body per section (no size cap). Task 4 replaces it with token-bounded packing. Tests here use small inputs that fit in one chunk.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `backend/tests/test_chunking.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_markdown'`.

- [ ] **Step 3: Rewrite `backend/app/chunking.py`**

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tiktoken
from markdown_it import MarkdownIt

_MD = MarkdownIt("commonmark").enable("table")


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    token_count: int
    content_hash: str
    chunk_id: str
    heading_path: list[str]


@dataclass
class _Block:
    kind: str  # "heading" | "table" | "fence" | "other"
    text: str
    heading_level: int | None = None
    heading_text: str | None = None


@dataclass
class _Section:
    heading_path: list[str]
    blocks: list[_Block]


def _make_id(source: str, chunk_index: int, content_hash: str) -> str:
    raw = f"{source}::{chunk_index}::{content_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _parse_blocks(md_text: str) -> list[_Block]:
    """Parse Markdown into top-level blocks using markdown-it token line maps."""
    tokens = _MD.parse(md_text)
    lines = md_text.split("\n")
    blocks: list[_Block] = []
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        is_leaf = tok.type in ("fence", "hr", "html_block")
        if tok.level != 0 or not (tok.type.endswith("_open") or is_leaf) or tok.map is None:
            i += 1
            continue
        start, end = tok.map
        raw = "\n".join(lines[start:end]).strip()
        if tok.type == "heading_open":
            level = int(tok.tag[1])  # "h2" -> 2
            heading_text = tokens[i + 1].content.strip() if i + 1 < n else ""
            blocks.append(_Block("heading", raw, level, heading_text))
        elif tok.type == "table_open":
            blocks.append(_Block("table", raw))
        elif tok.type == "fence":
            blocks.append(_Block("fence", raw))
        elif raw:
            blocks.append(_Block("other", raw))
        i += 1
    return blocks


def _group_sections(blocks: list[_Block]) -> list[_Section]:
    """Group content blocks by their heading breadcrumb path."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current: _Section | None = None
    for b in blocks:
        if b.kind == "heading":
            while stack and stack[-1][0] >= (b.heading_level or 1):
                stack.pop()
            stack.append((b.heading_level or 1, b.heading_text or ""))
            current = _Section([t for _, t in stack], [])
            sections.append(current)
        else:
            if current is None:
                current = _Section([], [])
                sections.append(current)
            current.blocks.append(b)
    non_empty = [s for s in sections if s.blocks]
    return non_empty or sections


def _breadcrumb(heading_path: list[str]) -> str:
    return " > ".join(heading_path) if heading_path else ""


def _with_breadcrumb(heading_path: list[str], body: str) -> str:
    crumb = _breadcrumb(heading_path)
    return f"{crumb}\n\n{body}" if crumb else body


def _section_bodies(
    section: _Section,
    enc: "tiktoken.Encoding",
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    # NAIVE: one body per section. Task 4 replaces this with token-bounded packing.
    body = "\n\n".join(b.text for b in section.blocks).strip()
    return [body] if body else []


def chunk_markdown(
    text: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    enc = tiktoken.get_encoding(encoding_name)
    sections = _group_sections(_parse_blocks(text))

    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        for body in _section_bodies(section, enc, chunk_size, chunk_overlap):
            chunk_text_ = _with_breadcrumb(section.heading_path, body)
            content_hash = hashlib.sha256(chunk_text_.encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    text=chunk_text_,
                    source=source,
                    chunk_index=idx,
                    token_count=len(enc.encode(chunk_text_)),
                    content_hash=content_hash,
                    chunk_id=_make_id(source, idx, content_hash),
                    heading_path=list(section.heading_path),
                )
            )
            idx += 1
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunking.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/chunking.py backend/tests/test_chunking.py
git commit -m "feat: markdown-structure-aware chunker with heading breadcrumbs"
```

---

### Task 4: Token-bounded packing with overlap

**Files:**
- Modify: `backend/app/chunking.py` (replace `_section_bodies`, add `_hard_split`, `_overlap_tail`)
- Modify: `backend/tests/test_chunking.py` (add tests)

**Interfaces:**
- Consumes: `_Section`, `_Block`, `_MD`, `Chunk` from Task 3.
- Produces: token-bounded `_section_bodies` — each returned body (plus its breadcrumb prefix) fits within `chunk_size` tokens where possible, with `chunk_overlap` tokens of carried context between bodies of the same section.

- [ ] **Step 1: Add the failing tests**

Append to `backend/tests/test_chunking.py`:

```python
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


def test_hash_inside_code_fence_is_not_a_heading():
    md = "# Real\n\n```python\n# this is a comment, not a heading\nx = 1\n```\n"
    chunks = chunk_markdown(md, source="code.md", chunk_size=500, chunk_overlap=10)
    # Only one heading path exists; the '#' comment must not create a section.
    paths = {tuple(c.heading_path) for c in chunks}
    assert paths == {("Real",)}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunking.py -v`
Expected: `test_oversized_section_splits_with_overlap` FAILS (naive `_section_bodies` returns one oversized body). `test_hash_inside_code_fence_is_not_a_heading` should already PASS (parser handles fences).

- [ ] **Step 3: Replace `_section_bodies` and add helpers in `chunking.py`**

Replace the entire `_section_bodies` function with the following, and add the two helper functions above it:

```python
def _hard_split(text: str, enc: "tiktoken.Encoding", budget: int) -> list[str]:
    toks = enc.encode(text)
    return [enc.decode(toks[i : i + budget]) for i in range(0, len(toks), budget)]


def _overlap_tail(body: str, enc: "tiktoken.Encoding", chunk_overlap: int) -> tuple[list[str], int]:
    if chunk_overlap <= 0:
        return [], 0
    toks = enc.encode(body)
    tail = toks[-chunk_overlap:]
    return [enc.decode(tail)], len(tail)


def _section_bodies(
    section: _Section,
    enc: "tiktoken.Encoding",
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    # Reserve token room for the breadcrumb prefix that will be prepended later.
    crumb = _breadcrumb(section.heading_path)
    prefix_tokens = len(enc.encode(f"{crumb}\n\n")) if crumb else 0
    budget = max(chunk_size - prefix_tokens, 1)

    # Expand any block that is itself larger than the budget into sub-units.
    units: list[str] = []
    for b in section.blocks:
        if len(enc.encode(b.text)) <= budget:
            units.append(b.text)
        else:
            units.extend(_hard_split(b.text, enc, budget))

    bodies: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for u in units:
        utoks = len(enc.encode(u))
        if cur and cur_tokens + utoks > budget:
            bodies.append("\n\n".join(cur).strip())
            cur, cur_tokens = _overlap_tail(bodies[-1], enc, chunk_overlap)
        cur.append(u)
        cur_tokens += utoks
    if cur:
        joined = "\n\n".join(cur).strip()
        if joined:
            bodies.append(joined)
    return bodies
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunking.py -v`
Expected: all pass (8+).

- [ ] **Step 5: Commit**

```bash
git add backend/app/chunking.py backend/tests/test_chunking.py
git commit -m "feat: token-bounded chunk packing with overlap"
```

---

### Task 5: Atomic table splitting

**Files:**
- Modify: `backend/app/chunking.py` (add `_split_table`, wire into `_section_bodies`)
- Modify: `backend/tests/test_chunking.py` (add test)

**Interfaces:**
- Consumes: `_section_bodies`, `_hard_split` from Task 4.
- Produces: oversized `table` blocks split by rows, re-emitting the header + separator rows in each part.

- [ ] **Step 1: Add the failing test**

Append to `backend/tests/test_chunking.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking.py::test_large_table_splits_by_rows_repeating_header -v`
Expected: FAIL — a big table currently goes through `_hard_split`, so only the first part keeps the header.

- [ ] **Step 3: Add `_split_table` and wire it in**

Add this function above `_section_bodies` in `chunking.py`:

```python
def _split_table(table_md: str, enc: "tiktoken.Encoding", budget: int) -> list[str]:
    rows = [r for r in table_md.split("\n") if r.strip()]
    if len(rows) < 2:
        return _hard_split(table_md, enc, budget)
    header, sep, body_rows = rows[0], rows[1], rows[2:]
    head = f"{header}\n{sep}"
    parts: list[str] = []
    cur = [header, sep]
    cur_tokens = len(enc.encode(head))
    for r in body_rows:
        rtoks = len(enc.encode(r))
        if len(cur) > 2 and cur_tokens + rtoks > budget:
            parts.append("\n".join(cur))
            cur = [header, sep]
            cur_tokens = len(enc.encode(head))
        cur.append(r)
        cur_tokens += rtoks
    if len(cur) > 2:
        parts.append("\n".join(cur))
    return parts or [table_md]
```

In `_section_bodies`, change the block-expansion loop so tables use `_split_table`:

```python
    for b in section.blocks:
        if len(enc.encode(b.text)) <= budget:
            units.append(b.text)
        elif b.kind == "table":
            units.extend(_split_table(b.text, enc, budget))
        else:
            units.extend(_hard_split(b.text, enc, budget))
```

- [ ] **Step 4: Run the full chunker suite**

Run: `pytest tests/test_chunking.py -v`
Expected: all pass (9+).

- [ ] **Step 5: Commit**

```bash
git add backend/app/chunking.py backend/tests/test_chunking.py
git commit -m "feat: split oversized markdown tables by rows, repeating header"
```

---

### Task 6: Rewire the documents route

**Files:**
- Modify: `backend/app/routes/documents.py`
- Modify: `backend/tests/test_api.py` (add upload tests)

**Interfaces:**
- Consumes: `get_converter`, `SUPPORTED_EXTENSIONS` (Task 2); `chunk_markdown`, `Chunk.heading_path` (Tasks 3–5).
- Produces: `POST /documents/upload` accepting the 8 supported types; Chroma metadata now includes `heading_path`.

- [ ] **Step 1: Add failing API tests**

Append to `backend/tests/test_api.py` (keep existing imports; add `import app.routes.documents as docs_mod` and `import pytest` if not present):

```python
import app.routes.documents as docs_mod


@pytest.fixture()
def _mock_ingest(monkeypatch):
    async def fake_embed(texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    async def fake_upsert(**kwargs):
        return None

    monkeypatch.setattr(docs_mod, "embed_texts", fake_embed)
    monkeypatch.setattr(docs_mod, "upsert_chunks", fake_upsert)


async def test_upload_csv_ok(client, _mock_ingest):
    files = {"file": ("people.csv", b"name,role\nAlice,eng\nBob,pm\n", "text/csv")}
    r = await client.post("/documents/upload", files=files)
    assert r.status_code == 201
    body = r.json()
    assert body["source"] == "people.csv"
    assert body["chunk_count"] >= 1


async def test_upload_unsupported_extension_rejected(client):
    files = {"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")}
    r = await client.post("/documents/upload", files=files)
    assert r.status_code == 415


async def test_upload_too_large_rejected(client):
    big = b"x" * (10 * 1024 * 1024 + 1)
    files = {"file": ("big.txt", big, "text/plain")}
    r = await client.post("/documents/upload", files=files)
    assert r.status_code == 413


async def test_upload_corrupt_pdf_returns_422(client, _mock_ingest):
    files = {"file": ("broken.pdf", b"this is not a real pdf", "application/pdf")}
    r = await client.post("/documents/upload", files=files)
    assert r.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: the four new tests FAIL (route still uses `_extract_pdf_text`; `_mock_ingest` patches names that will move).

- [ ] **Step 3: Rewrite `backend/app/routes/documents.py`**

Replace the whole file with:

```python
from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.chunking import chunk_markdown
from app.config import settings
from app.conversion import SUPPORTED_EXTENSIONS, get_converter
from app.llm import embed_texts
from app.schemas import IngestResponse, IngestTextRequest
from app.vectorstore import upsert_chunks

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


async def _ingest_text(text: str, source: str) -> IngestResponse:
    chunks = chunk_markdown(
        text=text,
        source=source,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No text content found",
        )

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
            "heading_path": " > ".join(c.heading_path),
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

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
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

    return await _ingest_text(text=markdown, source=filename)
```

- [ ] **Step 4: Run the API suite**

Run: `pytest tests/test_api.py -v`
Expected: all pass, including the four new upload tests.

- [ ] **Step 5: Run the full backend suite**

Run: `pytest -v`
Expected: all pass (chunking + conversion + api).

- [ ] **Step 6: Commit**

```bash
git add backend/app/routes/documents.py backend/tests/test_api.py
git commit -m "feat: route uploads through MarkItDown + markdown chunker"
```

---

### Task 7: Frontend accept list and documentation

**Files:**
- Modify: `frontend/src/App.tsx:125,129`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`

**Interfaces:**
- Consumes: `SUPPORTED_EXTENSIONS` list (mirrored as UI copy; no runtime import).

- [ ] **Step 1: Widen the file input in `App.tsx`**

Change the label text (line ~125) and the `accept` attribute (line ~129):

```tsx
            Upload File (.pdf / .docx / .xlsx / .xls / .pptx / .csv / .txt / .md)
```

```tsx
              accept=".pdf,.docx,.xlsx,.xls,.pptx,.csv,.txt,.md"
```

- [ ] **Step 2: Update `README.md`**

Replace the `/documents/upload` row description and any "(.txt / .md / .pdf)" mention so supported types read: `.pdf, .docx, .xlsx, .xls, .pptx, .csv, .txt, .md (converted to Markdown via MarkItDown)`.

- [ ] **Step 3: Update `ARCHITECTURE.md`**

a) Add a pointer near the top (after the intro line):

```markdown
> Ingestion redesign: see `docs/superpowers/specs/2026-07-18-markitdown-ingestion-design.md`.
```

b) In the "Backend file map", replace the `chunking.py` bullet and add a `conversion.py` bullet:

```markdown
- `backend/app/conversion.py` — `DocumentConverter` Protocol + `LocalMarkItDownConverter`: converts uploads to Markdown via MarkItDown, run in a thread pool with a timeout. `SUPPORTED_EXTENSIONS` is the single source of truth (pdf/docx/xlsx/xls/pptx/csv/txt/md). A future `RemoteMarkItDownConverter` can implement the same Protocol for out-of-process conversion.
- `backend/app/chunking.py` — `chunk_markdown()`: parses Markdown (markdown-it-py), splits on heading hierarchy, keeps tables/code intact, prefixes each chunk with its heading breadcrumb, then token-caps at `chunk_size`/`chunk_overlap`. Deterministic `chunk_id` unchanged.
```

c) In the `routes/documents.py` bullet, replace the PDF/pypdf description:

```markdown
  - `POST /documents/upload` validates the extension against `SUPPORTED_EXTENSIONS`, caps at 10 MB, converts to Markdown via `get_converter()`, then funnels into `_ingest_text()` (chunk → embed → upsert). Chroma metadata now includes `heading_path`.
```

d) Delete the "PDF text cleanup is heuristic" note under "Notable design details" (pypdf/`_extract_pdf_text` no longer exist).

- [ ] **Step 4: Verify frontend builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx README.md ARCHITECTURE.md
git commit -m "docs: widen upload types in UI/README, update ARCHITECTURE for MarkItDown ingestion"
```

---

## Self-Review

**Spec coverage:**
- §1 format scope → Task 1 (deps), Task 2 (`SUPPORTED_EXTENSIONS`). ✓
- §2 converter seam + thread-pool offload → Task 2. ✓
- §3 markdown-aware chunker (sections, breadcrumb prefix, table/code atomic, overlap, deterministic IDs) → Tasks 3–5. ✓
- §4 data flow (upload + paste both via `_ingest_text`/`chunk_markdown`) → Task 6. ✓
- §5 error handling (415/413/422/timeout) → Task 6 tests + route. ✓
- §6 dependencies → Task 1. ✓
- §7 frontend + docs (README + ARCHITECTURE incl. spec pointer) → Task 7. ✓
- §8 testing (rewritten chunking tests, new conversion tests, extended api tests) → Tasks 2–6. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `chunk_markdown(text, source, chunk_size, chunk_overlap)` and `Chunk.heading_path: list[str]` used identically in Tasks 3–6. `get_converter()`/`SUPPORTED_EXTENSIONS`/`to_markdown(data, filename)` consistent between Task 2 and Task 6. `_section_bodies(section, enc, chunk_size, chunk_overlap)` signature stable across Tasks 3–5. ✓

**Notes for the implementer:**
- `pytest-asyncio` is configured in the repo; the async API tests follow the existing `test_api.py` style (they rely on the project's asyncio mode). If a new async test is not collected, mark it with `@pytest.mark.asyncio`.
- The `+5` token tolerance in `test_oversized_section_splits_with_overlap` accounts for the breadcrumb prefix and decode boundaries; do not tighten it to exact equality.
