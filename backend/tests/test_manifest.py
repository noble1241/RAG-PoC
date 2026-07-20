import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

import json
from pathlib import Path

from app.manifest import record_ingestion


def _entry(chunk_ids, table_ids):
    return {
        "file_type": "csv",
        "ingested_at": "2026-07-20T00:00:00Z",
        "converted_markdown": "converted_output/x.csv.md",
        "chunk_ids": chunk_ids,
        "table_ids": table_ids,
    }


def test_record_creates_manifest(tmp_path: Path):
    p = tmp_path / "manifest.json"
    record_ingestion(p, "a.csv", _entry(["c1"], ["t1"]))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["a.csv"]["chunk_ids"] == ["c1"]
    assert data["a.csv"]["table_ids"] == ["t1"]


def test_record_is_incremental(tmp_path: Path):
    p = tmp_path / "manifest.json"
    record_ingestion(p, "a.csv", _entry(["c1"], []))
    record_ingestion(p, "b.csv", _entry(["c2"], ["t2"]))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"a.csv", "b.csv"}  # second add did not clobber first


def test_record_replaces_same_source(tmp_path: Path):
    p = tmp_path / "manifest.json"
    record_ingestion(p, "a.csv", _entry(["c1"], []))
    record_ingestion(p, "a.csv", _entry(["c1", "c2"], ["t9"]))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert list(data.keys()) == ["a.csv"]
    assert data["a.csv"]["chunk_ids"] == ["c1", "c2"]
    assert data["a.csv"]["table_ids"] == ["t9"]


def test_record_recovers_from_corrupt_manifest(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text("{ not valid json", encoding="utf-8")
    record_ingestion(p, "a.csv", _entry(["c1"], []))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["a.csv"]["chunk_ids"] == ["c1"]
