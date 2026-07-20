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
