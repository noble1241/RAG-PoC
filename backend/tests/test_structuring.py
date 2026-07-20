import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

from app.structuring import parse_markdown_table, render_table_summary


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
