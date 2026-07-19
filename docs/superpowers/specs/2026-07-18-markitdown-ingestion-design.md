# Design: MarkItDown-based ingestion with Markdown-aware chunking

Date: 2026-07-18
Status: Approved (design), pending implementation plan

## 1. Goal & guiding principle

Replace the current plain-text ingestion pipeline (pypdf + regex cleanup + blind
token-window chunking) with one that:

1. Converts any supported document to **structured Markdown** via Microsoft
   [MarkItDown](https://github.com/microsoft/markitdown).
2. Chunks that Markdown while **respecting its structure** (headings, tables,
   code, lists) instead of a position-blind token window.
3. Is built so the converter can move to a separate service later **without
   touching any caller** (isolation seam).

Motivation: the PoC needs to RAG over more than PDF/Word — specifically the
common **policy-document** formats — and MarkItDown's structured Markdown output
enables materially better chunking than the current blind 500-token window.

### Format scope (policy-document PoC)

| Category | Extensions | MarkItDown extra |
|---|---|---|
| PDF | `pdf` | `[pdf]` (pdfminer.six, pdfplumber) |
| Word | `docx` | `[docx]` (mammoth, lxml) |
| Excel | `xlsx`, `xls` | `[xlsx]`, `[xls]` (pandas, openpyxl, xlrd) |
| PowerPoint | `pptx` | `[pptx]` (python-pptx) |
| Simple text | `txt`, `md`, `csv` | core deps only |

**Explicitly out of scope for this PoC:** epub, msg, ipynb, json, xml, html,
images (OCR), audio (transcription), URL sources (YouTube/Wikipedia). All of
these convert fully **offline** — no external API key is needed for conversion,
which keeps tests deterministic.

## 2. Architecture

Pipeline becomes: **convert → chunk → embed → upsert**. `llm.py` (embed) and
`vectorstore.py` (upsert) are **unchanged**. Two modules change, one is new.

### New: `backend/app/conversion.py` (the isolation seam)

```python
class DocumentConverter(Protocol):
    async def to_markdown(self, data: bytes, filename: str) -> str: ...

class LocalMarkItDownConverter:
    # holds a single MarkItDown() instance
    async def to_markdown(self, data: bytes, filename: str) -> str:
        # loop.run_in_executor(None, ...) →
        #   md.convert_stream(io.BytesIO(data),
        #                     stream_info=StreamInfo(filename=filename))
        # wrapped in asyncio.wait_for(timeout) so a pathological file can't hang
```

- Conversion is CPU-bound and blocking → offloaded to the thread pool using the
  same `run_in_executor` pattern `vectorstore.py` already uses. This also fixes a
  latent bug today: `_extract_pdf_text` (pypdf) runs directly on the event loop,
  so a large PDF already blocks all other in-flight requests.
- A future `RemoteMarkItDownConverter` (HTTP call to a converter container)
  implements the same Protocol — a drop-in for production, no caller changes.
  This is the intended path when upload volume, multi-tenancy, or the desire to
  keep heavy parsing deps out of the API image justifies it.
- `SUPPORTED_EXTENSIONS` is defined here as the single source of truth and stays
  in sync with the installed MarkItDown extras (we never accept a type whose
  converter we did not install).

### Rewritten: `backend/app/chunking.py` (structure-aware — see §3)

Keeps the deterministic-ID scheme and the `Chunk` dataclass; adds a
`heading_path` field.

### Modified: `backend/app/routes/documents.py`

- Upload handler validates extension against `SUPPORTED_EXTENSIONS`, enforces the
  10 MB cap, calls `converter.to_markdown(...)`, then the new chunker.
- Deletes `_extract_pdf_text` and the pypdf import.
- The paste-text endpoint (`POST /documents`) skips conversion and feeds text
  straight into the chunker (treated as Markdown / plain text).

## 3. The Markdown-aware chunker (core of the rebuild)

Signature: `chunk_markdown(md, source, chunk_size, chunk_overlap) -> list[Chunk]`

**Parse, don't regex.** Use `markdown-it-py` to tokenize Markdown into blocks. A
naive line-splitter breaks on `#` inside fenced code blocks or `|` inside prose;
the parser handles these correctly.

**Build sections by heading hierarchy.** Walk the block tokens tracking the
current heading stack. Each heading opens a section holding the blocks beneath it
until the next heading of equal-or-higher level. Every section carries a
breadcrumb `heading_path`, e.g. `"Acme Leave Policy > Parental Leave > Notice
Period"`.

**Pack sections into chunks:**

- Section fits in `chunk_size` tokens (tiktoken `cl100k_base`, unchanged) → one
  chunk.
- Section too big → split at block boundaries (paragraphs, list items), greedily
  packing blocks up to the token cap, carrying `chunk_overlap` tokens of trailing
  context into the next chunk.
- Atomic blocks never split mid-structure:
  - A **table** larger than the cap is split by rows, re-emitting the header row
    in each part so every piece is a valid, self-describing table.
  - A **fenced code block** larger than the cap falls back to a hard token split
    (last resort).
- Tiny trailing sections (e.g. a lone heading) are merged forward into the next
  chunk to avoid useless fragments.

**Breadcrumb prefix for retrieval (decision: prefix into text).** Each chunk's
stored/embedded text is prefixed with its `heading_path`:

```
Acme Leave Policy > Parental Leave > Notice Period

<chunk body>
```

Rationale: policy documents are deeply hierarchical and the meaningful keywords
(e.g. "parental leave", "eligibility", "termination") often live in section
**titles** while the clause text underneath is generic. Prefixing the breadcrumb
means a query like "parental leave notice period" matches the section context
even when the body paragraph lacks those words, and the LLM sees where a quoted
chunk came from. Tradeoff: the stored chunk text is no longer byte-identical to
the source (a short label line is prepended); acceptable for a retrieval system.

**IDs & metadata.** `chunk_id = sha256("{source}::{index}::{content_hash}")[:32]`,
still deterministic → re-ingesting an unchanged source overwrites the same Chroma
rows instead of duplicating. Chroma metadata gains `heading_path`.

## 4. Data flow

```
Upload file
  → validate extension ∈ SUPPORTED_EXTENSIONS      (else 415)
  → size ≤ 10 MB                                    (else 413)
  → converter.to_markdown(bytes, filename)          [thread pool + timeout]  (fail → 422)
  → chunk_markdown(md) → structure-aware Chunks
  → embed_texts()                                   (unchanged)
  → upsert_chunks()                                 (unchanged)
  → IngestResponse

Paste text
  → chunk_markdown(text) directly → embed → upsert
```

## 5. Error handling

| Case | Response |
|---|---|
| Extension not in supported set | 415, listing supported types |
| File > 10 MB | 413 |
| MarkItDown raises (corrupt / encrypted file) | 422, logged via existing `logger.exception` |
| Conversion yields empty text | 422 "No text content found" (existing behavior) |
| Conversion exceeds timeout | 422 (or 504); ties into production resource-cap story |

The extension allowlist stays (defense-in-depth: never hand a disallowed type to
any converter), and MarkItDown's magika content-detection runs inside — so a
`.txt` that is actually a PDF is handled correctly. Strict upgrade over today's
name-only extension check.

## 6. Dependencies

- **Remove:** `pypdf`.
- **Add:** `markitdown[docx,pptx,xlsx,xls,pdf]` and `markdown-it-py`.
  - Granular extras (not `[all]`) deliberately exclude the heaviest optional deps
    (audio/SpeechRecognition, Azure, YouTube). Still pulls magika (→ onnxruntime),
    pdfminer.six, pdfplumber, mammoth, pandas, openpyxl, xlrd, lxml, python-pptx.
  - `tiktoken`, `tenacity`, `chromadb`, `openai` unchanged.
- Moderate image-size bump — the exact reason conversion sits behind a seam so it
  can later move out of the API image.

## 7. Frontend & docs (small)

- `frontend/src/App.tsx`: widen the file input `accept=` list and helper text.
  `frontend/src/api.ts` needs no change (multipart upload is format-agnostic).
- Optional: `IngestResponse` gains a detected `content_type` field for display.
  Low priority.
- Update `README.md` (supported types table) and `ARCHITECTURE.md` (new modules,
  flow, the converter seam).

## 8. Testing (fits existing pytest + conftest mocking)

**`test_chunking.py` (rewritten, offline):**

- Headings produce one section per heading; `heading_path` breadcrumbs correct.
- Oversized section splits at block boundaries with overlap.
- Table over cap splits by rows and repeats the header row.
- `#` inside a fenced code block does not start a new section.
- Deterministic IDs stable across two runs; different sources → disjoint IDs
  (keep the existing two tests).
- Empty input → `[]`.

**New `test_conversion.py` (offline, tiny real fixtures):** small `.csv`, `.docx`
(and optionally `.xlsx`, `.pptx`) fixtures → assert non-empty Markdown with
expected structure markers (e.g. table pipes from a CSV). No API keys, no network.

**`test_api.py` (extends existing, reuses `client` fixture + mocked
embed/upsert):** upload `.csv` → 201 with `chunk_count`; unsupported `.exe` →
415; oversize → 413; corrupt `.pdf` → 422.

## 9. Effort & sequencing (TDD)

1. `conversion.py` + `DocumentConverter` seam.
2. Rewrite `chunking.py` (`chunk_markdown`) + tests first (largest piece).
3. Rewire `routes/documents.py`.
4. Deps + test fixtures + API tests.
5. Frontend `accept` list + docs (`README.md`, `ARCHITECTURE.md`).

## 10. Non-goals / accepted risks (PoC)

- No image OCR, audio transcription, or URL ingestion.
- No separate converter microservice yet (seam designed for it; not built).
- Indirect prompt-injection via document content remains mitigated only by the
  existing system-prompt instruction in `llm.py` (unchanged).
- Conversion timeout guards against hangs but there is no per-file memory cap
  in-process (would come with the future microservice + container limits).
