"""COMPOSITION ADAPTER — turn a structured `PolicyDocument` (from
policy_extraction.py) into clean, well-labeled RAG chunks and ingest them
through the existing embed → upsert → manifest path.

Why: for grid-heavy policy documents, chunking the raw MarkItDown output gives
noisy chunks with meaningless breadcrumbs and empty `table_rows`. Extract first,
then RAG over the *extracted records* — one chunk per policy item, each carrying
a meaningful `service > category > item` breadcrumb and structured metadata.
Retrieval quality jumps and nothing downstream (embed/vectorstore/chat) changes.

This reuses the same `embed_texts`, `upsert_chunks`, `record_ingestion`, and
deterministic-ID scheme as the normal pipeline — it is just a different *source*
of chunks. `policy_to_chunks()` is pure and unit-testable.
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Any

from app.chunking import make_id
from app.config import settings
from app.llm import embed_texts
from app.manifest import record_ingestion
from app.policy_extraction import Category, Item, PolicyDocument, Service
from app.vectorstore import upsert_chunks

# app/ -> backend/ , to resolve a relative converted_output_dir predictably.
_BACKEND_DIR = Path(__file__).resolve().parents[1]


def _item_text(policy_name: str, service: Service, category: Category, item: Item) -> str:
    """One self-contained, human-readable record for a single policy line item."""
    lines = [
        f"{policy_name} > {service.service} > {category.category}",
        f"Item {item.item_number} - {item.expense_item}",
    ]
    if item.rule:
        lines.append(f"Rule: {item.rule}")
    lines.append(f"Approval required: {item.approval_required}")
    if item.approval_details:
        lines.append(f"Approval details: {item.approval_details}")
    if item.exception_conditions:
        lines.append(f"Exception conditions: {item.exception_conditions}")
    for label, values in (
        ("Condition", item.conditions),
        ("Note", item.notes),
        ("Location constraint", item.location_constraints),
        ("Vendor constraint", item.vendor_constraints),
        ("Required document", item.required_documents),
    ):
        for v in values:
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


def policy_to_chunks(policy: PolicyDocument, source: str) -> list[dict[str, Any]]:
    """Flatten a PolicyDocument into RAG chunk records:
    one policy-metadata chunk, one per service (guidelines/vendors), one per item.
    Returns dicts of {chunk_id, text, metadata}. Deterministic IDs → idempotent
    re-ingest. Pure function (no I/O)."""
    chunks: list[dict[str, Any]] = []
    idx = 0

    def add(text: str, meta: dict[str, Any]) -> None:
        nonlocal idx
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        chunks.append(
            {
                "chunk_id": make_id(source, idx, content_hash),
                "text": text,
                "metadata": {**meta, "source": source, "chunk_index": idx, "content_hash": content_hash},
            }
        )
        idx += 1

    m = policy.policy_metadata
    meta_lines = [policy.policy_name]
    if m.policy_description:
        meta_lines.append(m.policy_description)
    if m.effective_date:
        meta_lines.append(f"Effective date: {m.effective_date}")
    if m.benefits_period:
        meta_lines.append(f"Benefits period: {m.benefits_period}")
    meta_lines += [f"Note: {n}" for n in m.important_notes]
    add(
        "\n".join(meta_lines),
        {"content_type": "policy-metadata", "heading_path": policy.policy_name, "service": ""},
    )

    for s in policy.services:
        if s.operational_guidelines or s.vendor_constraints or s.required_documents:
            g = [f"{policy.policy_name} > {s.service}", f"Gross up: {s.gross_up}"]
            g += [f"Guideline: {x}" for x in s.operational_guidelines]
            g += [f"Vendor constraint: {x}" for x in s.vendor_constraints]
            g += [f"Required document: {x}" for x in s.required_documents]
            add(
                "\n".join(g),
                {"content_type": "policy-service", "heading_path": f"{policy.policy_name} > {s.service}",
                 "service": s.service},
            )
        for c in s.categories:
            for it in c.items:
                add(
                    _item_text(policy.policy_name, s, c, it),
                    {
                        "content_type": "policy-item",
                        "heading_path": f"{policy.policy_name} > {s.service} > {c.category}",
                        "service": s.service,
                        "category": c.category,
                        "category_code": c.category_code,
                        "item_number": it.item_number,
                        "expense_item": it.expense_item,
                        "approval_required": it.approval_required,
                    },
                )
    return chunks


def _manifest_path() -> Path:
    out_dir = Path(settings.converted_output_dir)
    if not out_dir.is_absolute():
        out_dir = _BACKEND_DIR / out_dir
    return out_dir / settings.manifest_filename


async def ingest_policy(
    policy: PolicyDocument, source: str, *, ingested_at: str | None = None
) -> dict[str, Any]:
    """Embed + upsert the policy's chunks into Chroma and record the manifest.
    Reuses the unchanged embed/upsert/manifest functions."""
    ingested_at = ingested_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
    chunks = policy_to_chunks(policy, source)

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{**c["metadata"], "ingested_at": ingested_at} for c in chunks]

    embeddings = await embed_texts(texts)
    await upsert_chunks(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    item_ids = [c["chunk_id"] for c in chunks if c["metadata"]["content_type"] == "policy-item"]
    record_ingestion(
        _manifest_path(),
        source,
        {
            "file_type": "policy-extraction",
            "ingested_at": ingested_at,
            "converted_markdown": None,
            "chunk_ids": ids,
            "table_ids": item_ids,  # policy-item chunks are the structured units here
        },
    )
    return {"source": source, "chunk_count": len(ids), "item_count": len(item_ids)}
