"""Unit tests for policy_to_chunks (pure composition, no I/O)."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.policy_extraction import Category, Item, PolicyDocument, PolicyMetadata, Service
from app.policy_rag import policy_to_chunks


def _sample() -> PolicyDocument:
    item = Item(
        item_number="51",
        expense_item="Lodging",
        rule="Up to 2 nights.",
        approval_required="Conditional",
        approval_details="Beyond $170/night needs approval.",
        conditions=["Max $170/night"],
    )
    category = Category(category="House Hunting", category_code="77", items=[item])
    service = Service(
        service="Travel Expenses",
        gross_up="Yes",
        operational_guidelines=["Book through the Employee Travel Center."],
        categories=[category],
    )
    return PolicyDocument(
        policy_name="Career Move",
        policy_metadata=PolicyMetadata(effective_date="January 2025", policy_description="IBT movers."),
        services=[service],
    )


def test_chunk_shape_and_counts():
    chunks = policy_to_chunks(_sample(), "PPG.xlsx [Career Move]")
    types = [c["metadata"]["content_type"] for c in chunks]
    assert types == ["policy-metadata", "policy-service", "policy-item"]


def test_item_chunk_carries_breadcrumb_and_metadata():
    chunks = policy_to_chunks(_sample(), "PPG.xlsx [Career Move]")
    item = next(c for c in chunks if c["metadata"]["content_type"] == "policy-item")
    assert item["text"].startswith("Career Move > Travel Expenses > House Hunting")
    assert "Item 51 - Lodging" in item["text"]
    assert item["metadata"]["item_number"] == "51"
    assert item["metadata"]["approval_required"] == "Conditional"


def test_ids_are_deterministic():
    a = policy_to_chunks(_sample(), "PPG.xlsx [Career Move]")
    b = policy_to_chunks(_sample(), "PPG.xlsx [Career Move]")
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]


def test_source_disambiguates_ids_across_policies():
    a = policy_to_chunks(_sample(), "PPG.xlsx [Career Move]")
    b = policy_to_chunks(_sample(), "PPG.xlsx [Company Request]")
    assert set(c["chunk_id"] for c in a).isdisjoint(c["chunk_id"] for c in b)


async def test_ingest_policy_wires_manifest_and_returns_counts(monkeypatch):
    import app.policy_rag as pr

    captured: dict = {}

    async def fake_embed(texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    async def fake_upsert(**kwargs):
        captured.update(kwargs)

    rec = MagicMock()
    monkeypatch.setattr(pr, "embed_texts", fake_embed)
    monkeypatch.setattr(pr, "upsert_chunks", fake_upsert)
    monkeypatch.setattr(pr, "record_ingestion", rec)

    result = await pr.ingest_policy(
        _sample(), source="PPG.xlsx [Career Move]", ingested_at="2026-07-21T00:00:00Z"
    )

    # return dict: 3 chunks total (metadata + service + item), 1 policy-item
    assert result["chunk_count"] == 3
    assert result["item_count"] == 1

    # manifest entry shape: record_ingestion(path, source, entry) — entry is positional arg 2
    entry = rec.call_args.args[2]
    assert entry["file_type"] == "policy-extraction"
    assert len(entry["table_ids"]) == 1          # the single policy-item chunk id
    assert entry["chunk_ids"] == captured["ids"] # all chunk ids recorded

    # ingested_at injected into every chunk's metadata
    assert all(m["ingested_at"] == "2026-07-21T00:00:00Z" for m in captured["metadatas"])
