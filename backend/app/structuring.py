from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from app.chunking import (
    Chunk,
    chunk_markdown,
    group_sections,
    make_id,
    parse_blocks,
    split_table,
)

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
