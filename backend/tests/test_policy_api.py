"""Route tests for POST /documents/extract-policy (LLM + Chroma mocked)."""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import AsyncMock

import pytest

import app.routes.policy as policy_mod
from app.policy_extraction import PolicyDocument, PolicyMetadata


def _doc(name: str) -> PolicyDocument:
    return PolicyDocument(policy_name=name, policy_metadata=PolicyMetadata())


@pytest.fixture()
def _redirect_output(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_mod.settings, "policy_output_dir", str(tmp_path))
    monkeypatch.setattr(policy_mod.settings, "converted_output_dir", str(tmp_path))
    return tmp_path


def _upload(text: bytes = b"policy body text"):
    return {"file": ("PPG.txt", BytesIO(text), "text/plain")}


async def test_extract_policy_multi_success(client, monkeypatch, _redirect_output):
    monkeypatch.setattr(policy_mod, "enumerate_policies", lambda md, **kw: ["Career Move", "Company Request"])
    monkeypatch.setattr(policy_mod, "extract_policy_per_service", lambda md, **kw: _doc(kw["extra_instructions"]))
    ingest = AsyncMock(return_value={"source": "x", "chunk_count": 3, "item_count": 2})
    monkeypatch.setattr(policy_mod, "ingest_policy", ingest)

    r = await client.post("/documents/extract-policy", files=_upload())
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "PPG.txt"
    assert len(body["policies"]) == 2
    assert body["errors"] == []
    assert ingest.await_count == 2
    # source disambiguates the two policies
    sources = {call.kwargs["source"] for call in ingest.await_args_list}
    assert sources == {"PPG.txt [Career Move]", "PPG.txt [Company Request]"}
    # JSON artifacts written
    assert (_redirect_output / "PPG.txt__Career_Move.json").exists()


async def test_extract_policy_partial_success(client, monkeypatch, _redirect_output):
    def flaky_extract(md, **kw):
        if "Company Request" in kw["extra_instructions"]:
            raise RuntimeError("truncated at output-token limit")
        return _doc(kw["extra_instructions"])

    monkeypatch.setattr(policy_mod, "enumerate_policies", lambda md, **kw: ["Career Move", "Company Request"])
    monkeypatch.setattr(policy_mod, "extract_policy_per_service", flaky_extract)
    monkeypatch.setattr(policy_mod, "ingest_policy", AsyncMock(return_value={"chunk_count": 1, "item_count": 1}))

    r = await client.post("/documents/extract-policy", files=_upload())
    assert r.status_code == 200
    body = r.json()
    assert len(body["policies"]) == 1
    assert len(body["errors"]) == 1
    assert body["errors"][0]["policy_name"] == "Company Request"


async def test_extract_policy_all_fail_returns_422(client, monkeypatch, _redirect_output):
    monkeypatch.setattr(policy_mod, "enumerate_policies", lambda md, **kw: ["A"])
    def boom(md, **kw):
        raise RuntimeError("refused")
    monkeypatch.setattr(policy_mod, "extract_policy_per_service", boom)
    monkeypatch.setattr(policy_mod, "ingest_policy", AsyncMock())

    r = await client.post("/documents/extract-policy", files=_upload())
    assert r.status_code == 422


async def test_extract_policy_no_policies_returns_422(client, monkeypatch, _redirect_output):
    monkeypatch.setattr(policy_mod, "enumerate_policies", lambda md, **kw: [])
    r = await client.post("/documents/extract-policy", files=_upload())
    assert r.status_code == 422


async def test_extract_policy_policy_filter_narrows(client, monkeypatch, _redirect_output):
    monkeypatch.setattr(policy_mod, "enumerate_policies", lambda md, **kw: ["Career Move", "Company Request"])
    monkeypatch.setattr(policy_mod, "extract_policy_per_service", lambda md, **kw: _doc(kw["extra_instructions"]))
    monkeypatch.setattr(policy_mod, "ingest_policy", AsyncMock(return_value={"chunk_count": 1, "item_count": 1}))

    r = await client.post("/documents/extract-policy?policy=career", files=_upload())
    assert r.status_code == 200
    body = r.json()
    assert len(body["policies"]) == 1
    assert body["policies"][0]["policy_name"] == "Career Move"


async def test_extract_policy_unsupported_extension_415(client):
    r = await client.post(
        "/documents/extract-policy",
        files={"file": ("bad.exe", BytesIO(b"MZ"), "application/octet-stream")},
    )
    assert r.status_code == 415


async def test_extract_policy_ingest_failure_recorded(client, monkeypatch, _redirect_output):
    monkeypatch.setattr(policy_mod, "enumerate_policies", lambda md, **kw: ["Career Move"])
    monkeypatch.setattr(policy_mod, "extract_policy_per_service", lambda md, **kw: _doc(kw["extra_instructions"]))
    failing = AsyncMock(side_effect=RuntimeError("chroma down"))
    monkeypatch.setattr(policy_mod, "ingest_policy", failing)

    r = await client.post("/documents/extract-policy", files=_upload())
    assert r.status_code == 422  # the only policy failed to ingest -> all failed
    body = r.json()
    assert body["detail"]  # error detail present
