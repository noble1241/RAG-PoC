import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

from app.structuring import parse_markdown_table, render_table_summary, structure_document


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


def test_oversized_table_split_respects_chunk_size_with_long_heading_path():
    # Regression test: split_budget must be derived from the REAL heading path
    # and REAL row count overhead, not from an empty-args render. A long,
    # non-trivial heading path plus a table too big for one chunk forces the
    # split_table(...) branch in structure_document.
    rows = "\n".join(f"| r{i} | v{i} |" for i in range(40))
    md = (
        "# A\n\n"
        "## B\n\n"
        "| name | value |\n"
        "| --- | --- |\n"
        f"{rows}\n"
    )
    doc = structure_document(
        markdown=md, source="big_table.docx", file_type="docx",
        ingested_at="2026-07-20T00:00:00Z", chunk_size=80, chunk_overlap=0,
    )
    assert len(doc.table_chunks) > 1  # forced the split branch
    for c in doc.table_chunks:
        assert c.token_count <= 80 + 5, f"chunk {c.chunk_index} has {c.token_count} tokens"
        # every part re-emits the header + separator so it stays a valid table
        assert "| name | value |" in c.text
        assert "| --- | --- |" in c.text
