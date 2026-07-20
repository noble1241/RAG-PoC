# Ingestion Structuring Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert an isolated structuring layer between MarkItDown conversion and the existing chunker that attaches a document-level metadata record, extracts tables into their own summary+rows chunks, and maintains an incremental JSON manifest — without touching conversion, embedding, or retrieval.

**Architecture:** A new pure module `app/structuring.py` reuses the existing tested `chunk_markdown` for narrative and its block parser for locating tables, so tables inherit correct heading paths. Tables become their own chunks in the same Chroma collection (deterministic summary header + rendered rows as embedded text; structured rows as JSON in metadata). A new `app/manifest.py` maintains `converted_output/manifest.json` incrementally. `routes/documents.py` is rewired to route through the layer and record the manifest; embed/upsert/retrieval stay unchanged.

**Tech Stack:** Python 3.11 (conda env RAG-env), FastAPI, MarkItDown, markdown-it-py, tiktoken, ChromaDB, pytest + pytest-asyncio. python-docx (test-only) for the DOCX fixture.

## Global Constraints

- **Do NOT change** the MarkItDown conversion call, the embedding model/provider (`llm.py`), the vector store (`vectorstore.py`), or retrieval (`routes/chat.py`). New behavior is additive/metadata-only.
- **Reuse** the existing `chunk_markdown`; do not rewrite it. Only add public aliases to `chunking.py` to expose its parser.
- Chunking config comes from settings: `chunk_size=500`, `chunk_overlap=50`. Token encoding: `cl100k_base` (tiktoken), cached offline.
- Tests must be **offline and deterministic** — mock OpenAI (`embed_texts`) and Chroma (`upsert_chunks`); no network. Keep all test/print output **ASCII** (Windows console is cp1252).
- Run tests from `backend/` with the RAG-env interpreter:
  `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest -q`
- Deterministic IDs: `chunk_id = sha256("{source}::{index}::{content_hash}")[:32]`; `content_hash = sha256(text)[:16]`. Re-ingest overwrites the same rows.
- Chroma metadata values must be `str | int | float | bool` — store table rows as a JSON **string**.

---

### Task 1: Table parsing + summary helpers (`structuring.py` pure functions)

**Files:**
- Create: `backend/app/structuring.py`
- Test: `backend/tests/test_structuring.py`

**Interfaces:**
- Consumes: nothing (pure, stdlib + re only).
- Produces:
  - `@dataclass TableChunk(text: str, rows: list[dict], source: str, chunk_index: int, token_count: int, content_hash: str, chunk_id: str, heading_path: list[str])`
  - `@dataclass StructuredDocument(metadata: dict, narrative_chunks: list, table_chunks: list[TableChunk])`
  - `parse_markdown_table(table_md: str) -> list[dict]`
  - `render_table_summary(heading_path: list[str], rows: list[dict], table_md: str) -> str`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_structuring.py`:

```python
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

from app.structuring import parse_markdown_table, render_table_summary


def test_parse_markdown_table_unescapes_and_maps_rows():
    # MarkItDown escapes underscores as "leave\_type"; keys must come back clean.
    tbl = (
        "| leave\\_type | annual\\_days |\n"
        "| --- | --- |\n"
        "| Parental | 42 |\n"
        "| Sick | 10 |"
    )
    rows = parse_markdown_table(tbl)
    assert rows == [
        {"leave_type": "Parental", "annual_days": "42"},
        {"leave_type": "Sick", "annual_days": "10"},
    ]


def test_render_table_summary_has_breadcrumb_summary_and_data():
    tbl = "| leave_type | annual_days |\n| --- | --- |\n| Parental | 42 |\n| Sick | 10 |"
    rows = parse_markdown_table(tbl)
    text = render_table_summary(["Policy", "Leave"], rows, tbl)
    assert text.startswith("Policy > Leave\n\n")
    assert "Table with columns: leave_type, annual_days (2 rows)." in text
    assert "| Parental | 42 |" in text  # row data still present for retrieval


def test_render_table_summary_without_heading_omits_breadcrumb():
    tbl = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    text = render_table_summary([], parse_markdown_table(tbl), tbl)
    assert text.startswith("Table with columns: a, b (1 rows).")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest tests/test_structuring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.structuring'`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/structuring.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass

# Unescape backslash-escaped Markdown punctuation MarkItDown emits in cells
# (e.g. "leave\_type" -> "leave_type").
_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>])")


@dataclass
class TableChunk:
    text: str            # deterministic summary header + rendered rows (embedded)
    rows: list[dict]     # structured rows (stored as JSON in Chroma metadata)
    source: str
    chunk_index: int
    token_count: int
    content_hash: str
    chunk_id: str
    heading_path: list[str]


@dataclass
class StructuredDocument:
    metadata: dict                 # {source, file_type, ingested_at}
    narrative_chunks: list         # list[app.chunking.Chunk]
    table_chunks: list[TableChunk]


def _unescape(cell: str) -> str:
    return _ESCAPE_RE.sub(r"\1", cell).strip()


def _split_row(row: str) -> list[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [_unescape(c) for c in s.split("|")]


def _is_separator(row: str) -> bool:
    cells = [c.strip() for c in _split_row(row)]
    non_empty = [c for c in cells if c]
    return bool(non_empty) and all(re.fullmatch(r":?-+:?", c) for c in non_empty)


def parse_markdown_table(table_md: str) -> list[dict]:
    """Parse a Markdown pipe table into a list of {column: value} dicts.

    Note (PoC tradeoff): splits cells on '|' and does not handle escaped pipes
    (\\|) inside cells — rare in policy tables, acceptable here.
    """
    lines = [ln for ln in table_md.splitlines() if ln.strip()]
    if not lines:
        return []
    header = _split_row(lines[0])
    rows: list[dict] = []
    for ln in lines[1:]:
        if _is_separator(ln):
            continue
        cells = _split_row(ln)
        rows.append({key: (cells[i] if i < len(cells) else "") for i, key in enumerate(header)})
    return rows


def render_table_summary(heading_path: list[str], rows: list[dict], table_md: str) -> str:
    """Build the embedded text for a table chunk: an optional heading breadcrumb,
    a deterministic one-line summary (columns + row count, NO LLM call), then the
    rendered rows so retrieval still has the actual data to answer from."""
    cols = list(rows[0].keys()) if rows else []
    summary = f"Table with columns: {', '.join(cols)} ({len(rows)} rows)."
    parts: list[str] = []
    crumb = " > ".join(heading_path)
    if crumb:
        parts.append(crumb)
    parts.append(summary)
    parts.append(table_md.strip())
    return "\n\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest tests/test_structuring.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/structuring.py backend/tests/test_structuring.py
git commit -m "feat: table parsing + deterministic summary helpers for structuring layer"
```

---

### Task 2: `structure_document` orchestrator + expose chunker parser

**Files:**
- Modify: `backend/app/chunking.py` (add public aliases only — no logic change)
- Modify: `backend/app/structuring.py` (add `structure_document`)
- Test: `backend/tests/test_structuring.py` (extend)

**Interfaces:**
- Consumes from `chunking.py`: `parse_blocks`, `group_sections`, `split_table`, `make_id`, `chunk_markdown`, `Chunk` (aliases added this task).
- Produces: `structure_document(markdown: str, source: str, file_type: str, ingested_at: str, chunk_size: int, chunk_overlap: int, encoding_name: str = "cl100k_base") -> StructuredDocument`

- [ ] **Step 1: Add public aliases to `chunking.py`**

Append at the end of `backend/app/chunking.py` (after `chunk_markdown`):

```python
# Public aliases so the structuring layer (app.structuring) can reuse the same
# block parser + helpers. This keeps table heading-paths consistent with
# narrative chunks. No behavior change to the functions themselves.
parse_blocks = _parse_blocks
group_sections = _group_sections
split_table = _split_table
make_id = _make_id
```

- [ ] **Step 2: Write the failing test**

Append to `backend/tests/test_structuring.py`:

```python
from app.structuring import structure_document

_MD_POLICY = (
    "# Company Policy\n\n"
    "## Leave\n\n"
    "Employees accrue leave each month.\n\n"
    "| leave_type | annual_days |\n"
    "| --- | --- |\n"
    "| Parental | 42 |\n"
    "| Sick | 10 |\n\n"
    "## Pay\n\n"
    "Salaries are paid monthly.\n"
)


def _structure(md, source="policy.docx"):
    return structure_document(
        markdown=md, source=source, file_type="docx",
        ingested_at="2026-07-20T00:00:00Z", chunk_size=500, chunk_overlap=50,
    )


def test_narrative_splits_on_headings():
    doc = _structure(_MD_POLICY)
    paths = {tuple(c.heading_path) for c in doc.narrative_chunks}
    assert ("Company Policy", "Leave") in paths
    assert ("Company Policy", "Pay") in paths


def test_table_extracted_into_separate_chunk_with_rows():
    doc = _structure(_MD_POLICY)
    assert len(doc.table_chunks) == 1
    t = doc.table_chunks[0]
    assert t.heading_path == ["Company Policy", "Leave"]
    assert t.rows == [
        {"leave_type": "Parental", "annual_days": "42"},
        {"leave_type": "Sick", "annual_days": "10"},
    ]
    assert "Table with columns: leave_type, annual_days (2 rows)." in t.text


def test_table_not_inline_in_narrative():
    doc = _structure(_MD_POLICY)
    assert all("| Parental |" not in c.text for c in doc.narrative_chunks)


def test_metadata_record_attached():
    doc = _structure(_MD_POLICY)
    assert doc.metadata == {
        "source": "policy.docx", "file_type": "docx",
        "ingested_at": "2026-07-20T00:00:00Z",
    }


def test_table_only_document_yields_one_table_no_narrative():
    md = "## Leave\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    doc = _structure(md)
    assert doc.narrative_chunks == []
    assert len(doc.table_chunks) == 1
    assert doc.table_chunks[0].heading_path == ["Leave"]


def test_empty_input_yields_no_chunks():
    doc = _structure("   ")
    assert doc.narrative_chunks == []
    assert doc.table_chunks == []
    assert doc.metadata["source"] == "policy.docx"


def test_table_and_narrative_ids_are_disjoint_and_deterministic():
    a = _structure(_MD_POLICY)
    b = _structure(_MD_POLICY)
    a_ids = [c.chunk_id for c in a.narrative_chunks] + [c.chunk_id for c in a.table_chunks]
    b_ids = [c.chunk_id for c in b.narrative_chunks] + [c.chunk_id for c in b.table_chunks]
    assert a_ids == b_ids                 # deterministic
    assert len(a_ids) == len(set(a_ids))  # no collisions
```

- [ ] **Step 3: Run test to verify it fails**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest tests/test_structuring.py -v`
Expected: FAIL — `ImportError: cannot import name 'structure_document'`

- [ ] **Step 4: Write minimal implementation**

Add imports at the top of `backend/app/structuring.py` (below `from dataclasses import dataclass`):

```python
import hashlib

import tiktoken

from app.chunking import (
    Chunk,
    chunk_markdown,
    group_sections,
    make_id,
    parse_blocks,
    split_table,
)
```

Add at the end of `backend/app/structuring.py`:

```python
def structure_document(
    markdown: str,
    source: str,
    file_type: str,
    ingested_at: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str = "cl100k_base",
) -> StructuredDocument:
    """Split MarkItDown output into narrative chunks (via the existing
    chunk_markdown) and separate table chunks that carry their section's
    heading path. Pure function: no I/O, no embedding, no Chroma."""
    metadata = {"source": source, "file_type": file_type, "ingested_at": ingested_at}
    if not markdown or not markdown.strip():
        return StructuredDocument(metadata=metadata, narrative_chunks=[], table_chunks=[])

    blocks = parse_blocks(markdown)

    # Narrative = the whole document with table blocks removed, handed to the
    # existing chunker unchanged so it stays the single owner of sectioning.
    table_free_md = "\n\n".join(b.text for b in blocks if b.kind != "table")
    narrative: list[Chunk] = chunk_markdown(
        table_free_md, source, chunk_size, chunk_overlap, encoding_name
    ) if table_free_md.strip() else []

    enc = tiktoken.get_encoding(encoding_name)
    # Reserve budget for the summary line when splitting oversized tables.
    empty_summary_tokens = len(enc.encode(render_table_summary([], [], "")))
    split_budget = max(chunk_size - empty_summary_tokens, 1)

    table_chunks: list[TableChunk] = []
    idx = len(narrative)  # continue the document's index space after narrative
    for sec in group_sections(blocks):
        for b in sec.blocks:
            if b.kind != "table":
                continue
            single = render_table_summary(sec.heading_path, parse_markdown_table(b.text), b.text)
            parts = [b.text] if len(enc.encode(single)) <= chunk_size else split_table(b.text, enc, split_budget)
            for part in parts:
                rows = parse_markdown_table(part)
                text = render_table_summary(sec.heading_path, rows, part)
                content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                table_chunks.append(
                    TableChunk(
                        text=text,
                        rows=rows,
                        source=source,
                        chunk_index=idx,
                        token_count=len(enc.encode(text)),
                        content_hash=content_hash,
                        chunk_id=make_id(source, idx, content_hash),
                        heading_path=list(sec.heading_path),
                    )
                )
                idx += 1
    return StructuredDocument(metadata=metadata, narrative_chunks=narrative, table_chunks=table_chunks)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest tests/test_structuring.py tests/test_chunking.py -v`
Expected: PASS (all structuring tests + the 13 existing chunking tests still green)

- [ ] **Step 6: Commit**

```bash
git add backend/app/chunking.py backend/app/structuring.py backend/tests/test_structuring.py
git commit -m "feat: structure_document orchestrator reusing chunk_markdown + table extraction"
```

---

### Task 3: Incremental JSON manifest (`manifest.py`)

**Files:**
- Create: `backend/app/manifest.py`
- Test: `backend/tests/test_manifest.py`

**Interfaces:**
- Produces: `record_ingestion(manifest_path: Path, source: str, entry: dict) -> None`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_manifest.py`:

```python
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

import json
from pathlib import Path

from app.manifest import record_ingestion


def _entry(chunk_ids, table_ids):
    return {
        "file_type": "csv",
        "ingested_at": "2026-07-20T00:00:00Z",
        "converted_markdown": "converted_output/x.csv.md",
        "chunk_ids": chunk_ids,
        "table_ids": table_ids,
    }


def test_record_creates_manifest(tmp_path: Path):
    p = tmp_path / "manifest.json"
    record_ingestion(p, "a.csv", _entry(["c1"], ["t1"]))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["a.csv"]["chunk_ids"] == ["c1"]
    assert data["a.csv"]["table_ids"] == ["t1"]


def test_record_is_incremental(tmp_path: Path):
    p = tmp_path / "manifest.json"
    record_ingestion(p, "a.csv", _entry(["c1"], []))
    record_ingestion(p, "b.csv", _entry(["c2"], ["t2"]))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"a.csv", "b.csv"}  # second add did not clobber first


def test_record_replaces_same_source(tmp_path: Path):
    p = tmp_path / "manifest.json"
    record_ingestion(p, "a.csv", _entry(["c1"], []))
    record_ingestion(p, "a.csv", _entry(["c1", "c2"], ["t9"]))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert list(data.keys()) == ["a.csv"]
    assert data["a.csv"]["chunk_ids"] == ["c1", "c2"]
    assert data["a.csv"]["table_ids"] == ["t9"]


def test_record_recovers_from_corrupt_manifest(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text("{ not valid json", encoding="utf-8")
    record_ingestion(p, "a.csv", _entry(["c1"], []))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["a.csv"]["chunk_ids"] == ["c1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.manifest'`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/manifest.py`:

```python
from __future__ import annotations

import json
import threading
from pathlib import Path

# Guards the read-modify-write below. Tradeoff (PoC): this is a single-process
# in-memory lock, so it does NOT protect against two processes writing the same
# manifest concurrently. Acceptable for the single-process PoC; a real system
# would use a DB or file lock.
_LOCK = threading.Lock()


def record_ingestion(manifest_path: Path, source: str, entry: dict) -> None:
    """Upsert one source's entry in the JSON manifest, incrementally.

    Keyed by source filename; re-ingesting a source replaces only its entry and
    leaves the rest untouched. A corrupt/absent manifest is treated as empty so
    a bad file never blocks new ingestions.
    """
    manifest_path = Path(manifest_path)
    with _LOCK:
        data: dict = {}
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except (json.JSONDecodeError, OSError):
                data = {}
        data[source] = entry
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest tests/test_manifest.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/manifest.py backend/tests/test_manifest.py
git commit -m "feat: incremental JSON manifest for ingested sources"
```

---

### Task 4: Rewire `routes/documents.py` through the structuring layer

**Files:**
- Modify: `backend/app/config.py` (add `manifest_filename` setting)
- Modify: `backend/app/routes/documents.py` (route through `structure_document`, embed+upsert both chunk types, add metadata fields, record manifest)
- Test: `backend/tests/test_api.py` (extend)

**Interfaces:**
- Consumes: `structure_document` (Task 2), `record_ingestion` (Task 3).
- Produces: unchanged `IngestResponse` (now `chunk_count` = narrative + table chunks). Chroma metadata now includes `content_type` on every chunk and `table_rows` (JSON string) on table chunks.

- [ ] **Step 1: Add the manifest filename setting**

In `backend/app/config.py`, after the `converted_output_dir` line (currently line 37), add:

```python
    manifest_filename: str = "manifest.json"
```

- [ ] **Step 2: Write the failing tests**

Append to `backend/tests/test_api.py`:

```python
async def test_upload_table_has_content_type_and_rows_metadata(client, monkeypatch):
    captured: dict = {}

    async def fake_embed(texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    async def fake_upsert(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(docs_mod, "embed_texts", fake_embed)
    monkeypatch.setattr(docs_mod, "upsert_chunks", fake_upsert)

    files = {"file": ("leave.csv", b"leave_type,annual_days\nParental,42\nSick,10\n", "text/csv")}
    r = await client.post("/documents/upload", files=files)
    assert r.status_code == 201

    metas = captured["metadatas"]
    assert all("content_type" in m for m in metas)
    table_meta = next(m for m in metas if m["content_type"] == "table-summary")
    rows = json.loads(table_meta["table_rows"])
    assert {"leave_type": "Parental", "annual_days": "42"} in rows


async def test_upload_updates_manifest(client, _mock_ingest, monkeypatch, tmp_path):
    monkeypatch.setattr(docs_mod.settings, "converted_output_dir", str(tmp_path))
    files = {"file": ("leave.csv", b"leave_type,annual_days\nParental,42\n", "text/csv")}
    r = await client.post("/documents/upload", files=files)
    assert r.status_code == 201

    manifest = tmp_path / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert "leave.csv" in data
    assert len(data["leave.csv"]["table_ids"]) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest tests/test_api.py -k "content_type or manifest" -v`
Expected: FAIL — `content_type` missing from metadata / `manifest.json` not created.

- [ ] **Step 4: Rewrite the ingestion route**

Replace the top imports of `backend/app/routes/documents.py` — add `import json` and the two new modules. The import block (lines 1-18) becomes:

```python
from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.artifacts import dump_markdown
from app.config import settings
from app.conversion import SUPPORTED_EXTENSIONS, get_converter
from app.llm import embed_texts
from app.manifest import record_ingestion
from app.schemas import IngestResponse, IngestTextRequest
from app.structuring import structure_document
from app.vectorstore import upsert_chunks
```

Replace the whole `_ingest_text` function (currently lines 30-76) with:

```python
def _resolve_converted_dir() -> Path:
    out_dir = Path(settings.converted_output_dir)
    if not out_dir.is_absolute():
        out_dir = _BACKEND_DIR / out_dir
    return out_dir


async def _ingest_text(
    text: str,
    source: str,
    file_type: str,
    converted_markdown: str | None = None,
) -> IngestResponse:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    doc = structure_document(
        markdown=text,
        source=source,
        file_type=file_type,
        ingested_at=ts,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not doc.narrative_chunks and not doc.table_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No text content found",
        )

    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for c in doc.narrative_chunks:
        texts.append(c.text)
        ids.append(c.chunk_id)
        metadatas.append(
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "ingested_at": ts,
                "content_hash": c.content_hash,
                "heading_path": " > ".join(c.heading_path),
                "content_type": "narrative",
            }
        )
    for c in doc.table_chunks:
        texts.append(c.text)
        ids.append(c.chunk_id)
        metadatas.append(
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "ingested_at": ts,
                "content_hash": c.content_hash,
                "heading_path": " > ".join(c.heading_path),
                "content_type": "table-summary",
                "table_rows": json.dumps(c.rows),
            }
        )

    embeddings = await embed_texts(texts)
    await upsert_chunks(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    # Manifest update (best-effort — a failure here must never break ingestion).
    try:
        record_ingestion(
            _resolve_converted_dir() / settings.manifest_filename,
            source,
            {
                "file_type": file_type,
                "ingested_at": ts,
                "converted_markdown": converted_markdown,
                "chunk_ids": [c.chunk_id for c in doc.narrative_chunks],
                "table_ids": [c.chunk_id for c in doc.table_chunks],
            },
        )
    except Exception:
        logger.exception("Failed to update manifest for %s", source)

    total_tokens = sum(c.token_count for c in doc.narrative_chunks) + sum(
        c.token_count for c in doc.table_chunks
    )
    doc_id = hashlib.sha256(source.encode()).hexdigest()[:16]
    logger.info(
        "Ingested %s: %d chunks (%d narrative, %d table), %d tokens",
        source,
        len(texts),
        len(doc.narrative_chunks),
        len(doc.table_chunks),
        total_tokens,
    )
    return IngestResponse(
        document_id=doc_id,
        source=source,
        chunk_count=len(texts),
        tokens_processed=total_tokens,
    )
```

Update the paste-text endpoint (currently lines 79-81) to pass `file_type`:

```python
@router.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(request: Request, body: IngestTextRequest) -> IngestResponse:
    return await _ingest_text(text=body.text, source=body.source, file_type="text")
```

Update the upload endpoint's dump-and-ingest tail (currently lines 121-133) to capture the converted path and pass `file_type`:

```python
    # Dump the raw converted Markdown to disk for inspection (best-effort — a
    # failure here must never break ingestion).
    converted_rel: str | None = None
    if settings.save_converted_markdown:
        try:
            saved = dump_markdown(markdown, filename, _resolve_converted_dir())
            converted_rel = (
                str(saved.relative_to(_BACKEND_DIR))
                if saved.is_relative_to(_BACKEND_DIR)
                else str(saved)
            )
            logger.info("Saved converted markdown to %s", saved)
        except Exception:
            logger.exception("Failed to save converted markdown for %s", filename)

    return await _ingest_text(
        text=markdown, source=filename, file_type=ext, converted_markdown=converted_rel
    )
```

- [ ] **Step 5: Run the full backend suite to verify pass**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest -q`
Expected: PASS — all existing tests plus the two new ones (the pre-existing `test_upload_csv_ok`, `test_ingest_text_success`, `test_upload_dumps_converted_markdown_to_disk` still green).

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/app/routes/documents.py backend/tests/test_api.py
git commit -m "feat: route ingestion through structuring layer + record manifest"
```

---

### Task 5: DOCX fixture + end-to-end structuring test

**Files:**
- Modify: `backend/requirements.txt` (add `python-docx` for test fixture generation)
- Test: `backend/tests/test_docx_fixture.py`

**Interfaces:**
- Consumes: `LocalMarkItDownConverter` (real, offline), `structure_document`, the `/documents/upload` route + manifest.

- [ ] **Step 1: Add the test-only dependency**

Append to `backend/requirements.txt`:

```
python-docx==1.1.2
```

Install it into RAG-env (this machine intercepts TLS — use the combined CA bundle noted in the project memory if pip fails):

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pip install python-docx==1.1.2`
Expected: `Successfully installed python-docx-1.1.2`

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_docx_fixture.py`:

```python
"""End-to-end structuring over a real generated .docx fixture (offline)."""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

import asyncio
import io
import json

import pytest
from unittest.mock import AsyncMock, patch

import app.routes.documents as docs_mod
from app.conversion import LocalMarkItDownConverter
from app.structuring import structure_document

docx = pytest.importorskip("docx", reason="python-docx required for the DOCX fixture")


def _make_docx_bytes() -> bytes:
    d = docx.Document()
    d.add_heading("Company Policy", level=1)
    d.add_heading("Leave", level=2)
    d.add_paragraph("Employees accrue leave each month.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "leave_type"
    table.rows[0].cells[1].text = "annual_days"
    for lt, ad in [("Parental", "42"), ("Sick", "10")]:
        cells = table.add_row().cells
        cells[0].text = lt
        cells[1].text = ad
    d.add_heading("Pay", level=2)
    d.add_paragraph("Salaries are paid monthly.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_docx_structuring_splits_headings_and_extracts_table():
    md = asyncio.run(
        LocalMarkItDownConverter(timeout_seconds=30).to_markdown(_make_docx_bytes(), "policy.docx")
    )
    doc = structure_document(
        markdown=md, source="policy.docx", file_type="docx",
        ingested_at="2026-07-20T00:00:00Z", chunk_size=500, chunk_overlap=50,
    )
    # 1. narrative split on the heading hierarchy
    paths = {tuple(c.heading_path) for c in doc.narrative_chunks}
    assert any("Leave" in p for p in paths)
    assert any("Pay" in p for p in paths)
    # 2. table rows extracted separately (not inline in narrative)
    assert len(doc.table_chunks) >= 1
    rows = doc.table_chunks[0].rows
    assert {"leave_type": "Parental", "annual_days": "42"} in rows
    assert all("| Parental |" not in c.text for c in doc.narrative_chunks)


@pytest.mark.asyncio
async def test_docx_upload_updates_manifest(client, monkeypatch, tmp_path):
    monkeypatch.setattr(docs_mod.settings, "converted_output_dir", str(tmp_path))
    with (
        patch.object(docs_mod, "embed_texts", new_callable=AsyncMock,
                     side_effect=lambda texts: [[0.0, 0.0, 0.0] for _ in texts]),
        patch.object(docs_mod, "upsert_chunks", new_callable=AsyncMock),
    ):
        files = {"file": ("policy.docx", _make_docx_bytes(),
                          "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        r = await client.post("/documents/upload", files=files)
    assert r.status_code == 201

    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "policy.docx" in data
    assert len(data["policy.docx"]["chunk_ids"]) >= 1
    assert len(data["policy.docx"]["table_ids"]) >= 1
```

- [ ] **Step 3: Run the test**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest tests/test_docx_fixture.py -v`
Expected: PASS (2 passed) — this is an end-to-end verification test over behavior built in Tasks 1-4, exercising the real MarkItDown DOCX path for the first time (no prior test converts a `.docx`). If it FAILS, that's a genuine DOCX-specific gap (most likely: mammoth did not map the heading styles to `#`/`##`, so `heading_path` is empty) — investigate before proceeding, do not paper over it. If the tests are SKIPPED, `python-docx` is not installed (Step 1 did not take) — install it and re-run so they actually execute.

- [ ] **Step 4: Run the full suite**

Run: `C:\Users\noble\miniconda3\envs\RAG-env\python.exe -m pytest -q`
Expected: PASS — entire backend suite green.

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/tests/test_docx_fixture.py
git commit -m "test: DOCX fixture end-to-end structuring + manifest assertions"
```

---

## Notes for the implementer

- **Do not** touch `llm.py`, `vectorstore.py`, `routes/chat.py`, `conversion.py` internals, or the frontend. If a change seems to require it, stop and flag it — that would violate a global constraint.
- `chat.py` reads only `source` + `chunk_index` from metadata and sends the chunk `document` text to the LLM. The new `content_type`/`table_rows` fields are additive; retrieval keeps working unchanged, and table answers still work because the row data lives in the embedded text.
- The existing `_split_table` path inside `chunk_markdown` remains for any caller that passes table-bearing Markdown directly; the upload + paste routes now strip tables out first, so in practice narrative chunks are table-free.
