from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tiktoken


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    token_count: int
    content_hash: str
    chunk_id: str  # deterministic: sha256(source + chunk_index + content_hash)[:16]


def _make_id(source: str, chunk_index: int, content_hash: str) -> str:
    raw = f"{source}::{chunk_index}::{content_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_tokens = token_ids[start:end]
        chunk_text_ = enc.decode(chunk_tokens)
        content_hash = hashlib.sha256(chunk_text_.encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                text=chunk_text_,
                source=source,
                chunk_index=idx,
                token_count=len(chunk_tokens),
                content_hash=content_hash,
                chunk_id=_make_id(source, idx, content_hash),
            )
        )
        if end == len(token_ids):
            break
        start = end - chunk_overlap
        idx += 1

    return chunks
