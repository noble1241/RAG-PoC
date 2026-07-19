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
        is_leaf = tok.type in ("fence", "hr", "html_block", "code_block")
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
    # Hard-split pieces are sized to leave headroom for the overlap tail that
    # gets prepended to the next packed body, so tail + unit still fits budget.
    split_budget = max(budget - chunk_overlap, 1)
    units: list[str] = []
    for b in section.blocks:
        if len(enc.encode(b.text)) <= budget:
            units.append(b.text)
        else:
            units.extend(_hard_split(b.text, enc, split_budget))

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
