# Design: Structuring layer between conversion and chunking

Date: 2026-07-20
Status: Approved (design), pending implementation plan

## 1. Goal & guiding principle

Insert a **structuring layer** between the existing MarkItDown conversion step and
the existing chunker so that downstream retrieval works off structured units
instead of unstructured blobs. Concretely: attach a document-level metadata
record, split tables out of narrative into their own structured chunks, and
maintain a manifest mapping sources to their chunks and tables.

**Hard constraints (from the request):**

- Do **not** change the MarkItDown call or the embedding model/provider.
- Do **not** rewrite retrieval logic — only *enrich* the metadata it sees.
- Keep the new structuring step an **isolated, independently unit-testable**
  function/module.

**Reconciliation with existing code (important).** As of 2026-07-18 this codebase
already shipped a Markdown-structure-aware pipeline (spec:
`2026-07-18-markitdown-ingestion-design.md`). That changes what "new" means here.
Mapping the request against reality:

| Request item | Current reality | This spec |
|---|---|---|
| 1. Doc-level metadata front matter | Per-chunk metadata exists; **no doc-level record**, no file-type field, no page count | Add doc-level record (`source`, `file_type`, `ingested_at`). **Page count skipped** — MarkItDown returns text only; obtaining it means re-parsing each source with extra libs, not worth it for the PoC. |
| 2. Heading-based chunking | **Already fully implemented** in `chunk_markdown` (splits on headings, size fallback, `heading_path` on every chunk) | Reuse as-is. No rework. |
| 3. Table extraction (separate structured store) | **Opposite decision shipped**: tables kept *inline* in narrative chunks (`_split_table`) | Override for the upload path: extract tables into their own chunks (see §3). |
| 4. Manifest / index | Does not exist | Add incremental JSON manifest (§4). |
| 5. Chunk metadata for retrieval | Has source, heading_path, chunk_index, content_hash, ingested_at; **missing `content_type`** | Add `content_type` + table `table_rows`. |

Decisions locked with the user:

- **Tables → Option 1 (extract + summary embed).** Tables become their own
  chunks in the same Chroma collection; each carries a **deterministic** summary
  (no LLM call) and its structured rows as JSON in metadata. Rejected: keeping
  tables inline (does not meet "store separately"); excluding tables from vector
  search (would require a new retrieval lookup path — violates "don't rewrite
  retrieval").
- **Scope → build only the new parts; reuse the tested `chunk_markdown`.** Do not
  rework the existing chunker (28 passing tests).
- **Manifest → JSON.** The project uses JSON throughout and has no CSV.

## 2. Architecture

Pipeline becomes: **convert → structure → embed → upsert**. `llm.py` (embed),
`vectorstore.py` (upsert), `chat.py` (retrieval), and the frontend are
**unchanged**.

```
Upload file
  → validate ext / size            (unchanged)
  → converter.to_markdown(...)      (unchanged — MarkItDown)
  → structure_document(md, source, file_type)   ← NEW isolated module
  → embed_texts(all chunk texts)    (unchanged)
  → upsert_chunks(...)              (unchanged — richer metadata only)
  → manifest.record(...)            ← NEW side file (best-effort)
```

### New: `backend/app/structuring.py` (the layer)

Pure function, no I/O, no embedding, no Chroma — so it unit-tests standalone:

```python
@dataclass
class TableChunk:
    text: str            # deterministic header + rendered rows (this is embedded)
    rows: list[dict]     # structured rows
    source: str
    chunk_index: int
    token_count: int
    content_hash: str
    chunk_id: str
    heading_path: list[str]

@dataclass
class StructuredDocument:
    metadata: dict            # {source, file_type, ingested_at}
    narrative_chunks: list[Chunk]        # from existing chunk_markdown
    table_chunks: list[TableChunk]

def structure_document(markdown: str, source: str, file_type: str,
                       chunk_size: int, chunk_overlap: int) -> StructuredDocument: ...
```

It reuses the existing block parser. `chunking.py` is refactored **only** to
expose its already-written internals (`_parse_blocks`, `_group_sections`) for
import — the packing/ID logic and public `chunk_markdown` signature are unchanged.
This gives table chunks the correct `heading_path` of the section they sit in.

### Modified: `backend/app/routes/documents.py`

`_ingest_text` calls `structure_document` instead of `chunk_markdown` directly,
then embeds and upserts **both** narrative and table chunks through the *same*
`embed_texts` / `upsert_chunks` calls (just more items), and records the manifest
entry. No change to embed or upsert internals.

### New: `backend/app/manifest.py`

Small module owning the incremental JSON manifest (§4). Best-effort, like the
existing markdown dump: failures log but never break ingestion.

## 3. Table extraction (Option 1: extract + deterministic summary embed)

The document is parsed once with the existing block parser to locate every table
block and the `heading_path` of the section it sits in. From that, two things are
derived:

- **Table-free Markdown** — the original Markdown with only the table blocks
  removed (headings, prose, lists, code all preserved). This is passed to the
  existing `chunk_markdown` **unchanged**, which re-derives the identical heading
  sections and emits narrative `Chunk`s tagged `content_type="narrative"`. Removing
  whole table blocks (not partitioning per section by hand) keeps `chunk_markdown`
  the single owner of narrative sectioning/packing — no logic is duplicated.
- **One `TableChunk` per table block**, carrying the `heading_path` captured for
  it during the parse:
  - **Parse** the Markdown pipe table into `rows: list[dict]` keyed by the header
    cells. MarkItDown escapes some chars (e.g. `leave\_type`); unescape for keys.
  - **Deterministic summary + data as the embedded `text`.** Critical: retrieval
    sends the chunk's `document` text to the LLM (`chat.py` unchanged), so the
    embedded text must contain the actual rows, not only a bare summary. Format:

    ```
    <heading breadcrumb>

    Table: <col_a> by <col_b> (<N> rows). Columns: <col_a>, <col_b>, ...

    | col_a | col_b |
    | --- | --- |
    | ... | ... |
    ```

    The descriptive first line gives the embedding real semantic hooks (so
    "parental leave days" matches a table whose body is just numbers); the
    rendered rows keep the answer retrievable. No LLM call — summary is built
    from the heading + column headers + row count.
  - **`rows` JSON** is stored in metadata (`table_rows`) as the structured bonus
    that enables a future structured lookup without any retrieval rewrite now.
  - Oversized tables (over `chunk_size`) split by rows, repeating the header +
    summary line per part — same principle the existing `_split_table` already
    uses, so each part stays a valid self-describing table chunk.

**Tradeoff (note in code):** embedding one summary+rows chunk per table (vs. one
embedding per row) keeps infra identical to today — same collection, same upsert.
Per-row embeddings would retrieve individual facts better but multiply vector
count and need dedup/aggregation at query time; excluded from vector search
entirely would need a new retrieval path. Option 1 is the least-infra choice that
still lets tables be *found*.

## 4. Manifest / index

Single file `backend/converted_output/manifest.json` (beside the existing
markdown dumps). Keyed by source filename; **incremental** — load, upsert this
source's entry, write back. Never rebuilt from scratch. Re-ingest replaces the
entry (consistent with idempotent deterministic-ID upserts).

```json
{
  "leave_days.xlsx": {
    "file_type": "xlsx",
    "ingested_at": "2026-07-20T14:03:11Z",
    "converted_markdown": "converted_output/leave_days.xlsx.md",
    "chunk_ids": ["a1b2...", "c3d4..."],
    "table_ids": ["f9e8..."]
  }
}
```

**Concurrency tradeoff (note in code):** read-modify-write is not safe under truly
concurrent ingests of different files. Guarded with a simple in-process lock;
acceptable for a single-process PoC. Manifest path is configurable via settings,
defaulting under `converted_output_dir`.

## 5. Metadata schema (enrichment only)

Every embedded chunk (narrative **and** table) carries, in Chroma metadata:

| Field | Narrative | Table | Status |
|---|---|---|---|
| `source` | ✓ | ✓ | existing |
| `chunk_index` | ✓ | ✓ | existing |
| `heading_path` | ✓ | ✓ | existing |
| `ingested_at` | ✓ | ✓ | existing |
| `content_hash` | ✓ | ✓ | existing |
| `content_type` | `"narrative"` | `"table-summary"` | **new** |
| `table_rows` | — | JSON string | **new** |

`chat.py` continues to read only `source` + `chunk_index` and send `content` to
the LLM — unchanged. New fields are available to retrieval but not required by it.

Doc-level metadata (`source`, `file_type`, `ingested_at`) lives on the
`StructuredDocument` and is threaded into the manifest — **not** embedded as YAML
text in any chunk.

## 6. Data flow (paste-text path)

`POST /documents` (paste text) also routes through `structure_document` with
`file_type="text"`, so a pasted Markdown table is extracted the same way. Its
manifest entry uses the `source` label; `converted_markdown` is null (no file
dump for pasted text).

## 7. Testing

Reuses existing pytest + `conftest` mocking (offline, no network, ASCII output).

**New `test_structuring.py` (offline, pure function):**

- Narrative splits on headings (delegates to `chunk_markdown`; assert heading
  paths on narrative chunks).
- A table under a heading is extracted into a **separate** `TableChunk` — its
  rows are a `list[dict]` with the right keys/values, and it is **not** present
  inline in any narrative chunk.
- Table chunk's embedded `text` contains both the summary line and the row data;
  `heading_path` matches the section.
- `content_type` is `"narrative"` vs `"table-summary"` correctly.
- Empty / no-table input → empty `table_chunks`, narrative unchanged.

**New `test_manifest.py`:** ingesting a source writes an entry with correct
`chunk_ids` + `table_ids`; a second source **adds** (incremental, not rebuilt);
re-ingesting the same source **replaces** its entry.

**Fixture:** a small `.docx` with a heading hierarchy **and** at least one table.
Convert it (real MarkItDown, offline) → assert via `structure_document`: chunks
split on headings, table rows extracted separately, manifest updated.

**`test_api.py` (extend):** upload the fixture → 201; assert the response and that
table + narrative chunks were upserted (via existing mocked `upsert`).

## 8. Non-goals / accepted risks (PoC)

- No per-row table embeddings; no structured (non-vector) table lookup path.
- No page count in doc metadata (would need per-format re-parsing).
- Manifest read-modify-write not safe across concurrent different-file ingests
  (in-process lock only).
- The existing `chunk_markdown` inline-table path (`_split_table`) remains for
  any caller that passes table-bearing Markdown straight to it, but the upload +
  paste paths now strip tables out before chunking, so in practice narrative
  chunks are table-free.
