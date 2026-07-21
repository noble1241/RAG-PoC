import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

from app.artifacts import dump_markdown


def test_dump_markdown_writes_file_named_after_source(tmp_path):
    path = dump_markdown("# Hello\n\nWorld", "report.pdf", tmp_path)
    assert path.name == "report.pdf.md"
    assert path.parent == tmp_path
    assert path.read_text(encoding="utf-8") == "# Hello\n\nWorld"


def test_dump_markdown_creates_missing_dir(tmp_path):
    nested = tmp_path / "converted_output"
    path = dump_markdown("x", "a.docx", nested)
    assert path.exists()
    assert path == nested / "a.docx.md"


def test_dump_markdown_sanitizes_path_traversal(tmp_path):
    path = dump_markdown("x", "../../etc/evil name.pdf", tmp_path)
    # Only the basename is used, unsafe chars replaced; stays inside out_dir.
    assert path.parent == tmp_path
    assert "/" not in path.name and "\\" not in path.name
    assert path.name == "evil_name.pdf.md"


def test_dump_markdown_overwrites_same_source(tmp_path):
    dump_markdown("first", "doc.txt", tmp_path)
    path = dump_markdown("second", "doc.txt", tmp_path)
    assert path.read_text(encoding="utf-8") == "second"


def test_dump_policy_json_writes_named_file(tmp_path):
    from app.artifacts import dump_policy_json

    path = dump_policy_json('{"policy_name": "Career Move"}', "PPG.xlsx", "Career Move", tmp_path)
    assert path == tmp_path / "PPG.xlsx__Career_Move.json"
    assert path.read_text(encoding="utf-8") == '{"policy_name": "Career Move"}'
