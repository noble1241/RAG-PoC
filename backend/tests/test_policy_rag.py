"""Unit tests for policy_to_chunks (pure composition, no I/O)."""
from __future__ import annotations

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
