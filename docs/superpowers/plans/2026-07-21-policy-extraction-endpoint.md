# Policy Extraction Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated `POST /documents/extract-policy` endpoint that extracts the same structured `PolicyDocument` JSON from any policy-document format (xlsx grid, pdf, docx, …), fans a multi-policy workbook out into one document per policy automatically, writes a JSON artifact per policy, and RAG-ingests each.

**Architecture:** LLM-driven enumeration (Approach 1). One "enumerate the distinct policies" call returns policy names; each name is then extracted in its own focused map-reduce pass via the existing `extract_policy_per_service`. A prose PDF enumerates to one policy and takes the identical path — no grid special-casing. The endpoint reuses the existing conversion, embed, upsert, and manifest seams unchanged; only the *source* of chunks and its enrichment are new.

**Tech Stack:** FastAPI (async), Pydantic + OpenAI structured outputs (`chat.completions.parse`), MarkItDown conversion, ChromaDB, pytest (offline, mocked).

## Global Constraints

- **Test interpreter (RAG-env conda, Python 3.11):** run tests from `backend/` with
  `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest`. `asyncio_mode = auto` (no `@pytest.mark.asyncio` needed, but existing tests use it — harmless).
- **Offline tests only:** OpenAI and Chroma are always mocked; no test performs network I/O. Keep test output ASCII (cp1252 console).
- **SSL (runtime only, not tests):** OpenAI calls on this machine need the combined CA bundle (`SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`) due to Avast TLS interception.
- **Schema stays relocation-PPG:** do not generalize `PolicyDocument`'s schema/prompt to other policy domains.
- **Reuse, don't rewrite:** `extract_policy_per_service`, `policy_to_chunks`, `ingest_policy`, `embed_texts`, `upsert_chunks`, `record_ingestion`, `get_converter`, `dump_markdown`, `SUPPORTED_EXTENSIONS`, and the 10 MB cap are all reused as-is.
- **Blocking LLM calls off the event loop:** `enumerate_policies` and `extract_policy_per_service` are synchronous (OpenAI sync client). In the async route they MUST be called via `await asyncio.to_thread(...)` so they never block the event loop. `ingest_policy` is already async — `await` it directly.

## File Structure

- **`backend/app/policy_extraction.py`** (modify) — add `PolicyNames` schema, `enumerate_policies()`, and the pure `scope_to_section()` helper. Existing extractors untouched.
- **`backend/app/policy_rag.py`** (unchanged source) — `policy_to_chunks` / `ingest_policy` reused. Gains a test file.
- **`backend/app/artifacts.py`** (modify) — add `dump_policy_json()`, reusing `_safe_name`.
- **`backend/app/config.py`** (modify) — add `policy_extraction_model`, `policy_output_dir`.
- **`backend/.env.example`** (modify) — document the two new settings.
- **`backend/app/schemas.py`** (modify) — add `PolicyExtractionResult`, `PolicyExtractionError`, `PolicyExtractionResponse`.
- **`backend/app/routes/policy.py`** (create) — the endpoint orchestrator.
- **`backend/app/main.py`** (modify) — register the new router.
- **Tests (create):** `tests/test_policy_extraction.py`, `tests/test_policy_rag.py`, `tests/test_policy_api.py`; append to `tests/test_artifacts.py`.
- **Docs (modify):** `ARCHITECTURE.md`, `README.md`.

---

### Task 1: Enumeration + section-scoping primitives (`policy_extraction.py`)

**Files:**
- Modify: `backend/app/policy_extraction.py`
- Test: `backend/tests/test_policy_extraction.py`

**Interfaces:**
- Consumes: existing `_parse(client, model, system, user, response_format)` and `settings` in `policy_extraction.py`; `OpenAI`, `BaseModel`, `Field` (already imported).
- Produces:
  - `class PolicyNames(BaseModel)` with `policies: list[str]`.
  - `enumerate_policies(markdown: str, *, model: str | None = None, client: OpenAI | None = None) -> list[str]`
  - `scope_to_section(markdown: str, heading: str) -> str`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_policy_extraction.py`:

```python
"""Unit tests for policy enumeration + section scoping (OpenAI mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.policy_extraction import PolicyNames, enumerate_policies, scope_to_section


def _mock_client(parsed: PolicyNames) -> MagicMock:
    """Build an OpenAI-like mock whose chat.completions.parse returns `parsed`."""
    client = MagicMock()
    msg = MagicMock(refusal=None, parsed=parsed)
    client.chat.completions.parse.return_value = MagicMock(choices=[MagicMock(message=msg)])
    return client


def test_enumerate_policies_returns_multiple():
    client = _mock_client(PolicyNames(policies=["Career Move", "Company Request"]))
    assert enumerate_policies("# grid", model="gpt-4o-mini", client=client) == [
        "Career Move",
        "Company Request",
    ]


def test_enumerate_policies_single_policy():
    client = _mock_client(PolicyNames(policies=["Relocation Policy"]))
    assert enumerate_policies("prose", model="gpt-4o-mini", client=client) == ["Relocation Policy"]


def test_scope_to_section_slices_one_section():
    md = "## IBT & IAM\nrow a\nrow b\n## Appendix\nappendix text\n"
    out = scope_to_section(md, "ibt")
    assert "row a" in out
    assert "row b" in out
    assert "appendix text" not in out


def test_scope_to_section_no_match_returns_full():
    md = "## Sheet1\nbody\n"
    assert scope_to_section(md, "does-not-exist") == md


def test_scope_to_section_case_insensitive_substring():
    md = "## IBT & IAM\nbody\n"
    assert scope_to_section(md, "iam").startswith("## IBT & IAM")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (from `backend/`): `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest tests/test_policy_extraction.py -q`
Expected: FAIL — `ImportError: cannot import name 'PolicyNames'` (and `enumerate_policies` / `scope_to_section`).

- [ ] **Step 3: Implement the primitives**

In `backend/app/policy_extraction.py`, add the schema next to the other helper schemas (after `class ServiceNames(...)`):

```python
class PolicyNames(BaseModel):
    policies: list[str] = Field(
        description="Distinct policy names / policy-column labels present, in document order."
    )
```

Add the enumeration system prompt near `_SYSTEM_PROMPT`:

```python
_ENUMERATE_SYSTEM = """\
You identify the distinct policies in a document converted to Markdown. A workbook \
grid may place several policies as side-by-side columns; a prose document usually \
contains exactly one policy.

Return the distinct policy names / policy-column labels, in document order. In a grid, \
use the label at the top of each policy column (e.g. 'Career Move', 'Company Request'). \
In a single-policy document, return exactly one name. Do not invent policies.
"""
```

Add the two functions (place `enumerate_policies` after `extract_policy_per_service`, and `scope_to_section` above the CLI runner):

```python
def enumerate_policies(
    markdown: str, *, model: str | None = None, client: OpenAI | None = None
) -> list[str]:
    """List the distinct policies present in a document's Markdown via one
    structured-output LLM call. A single-policy prose doc returns one name; a
    multi-policy grid returns one label per policy column."""
    client = client or OpenAI(api_key=settings.openai_api_key)
    model = model or settings.chat_model
    return _parse(client, model, _ENUMERATE_SYSTEM, markdown, PolicyNames).policies


def scope_to_section(markdown: str, heading: str) -> str:
    """Return only the Markdown under the first `## <heading>` section (case-insensitive
    substring match on the heading text). If no `## ` heading matches, return the
    Markdown unchanged so a bad filter never discards the whole document."""
    lines = markdown.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and heading.lower() in line[3:].strip().lower():
            start = i
            break
    if start is None:
        return markdown
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest tests/test_policy_extraction.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/policy_extraction.py backend/tests/test_policy_extraction.py
git commit -m "feat: add enumerate_policies + scope_to_section primitives"
```

---

### Task 2: Cover the composition adapter (`policy_to_chunks` tests)

Closes the pre-existing "new modules have no tests" gap. No source change — `policy_rag.py` is reused unchanged; this locks in its behavior before the route depends on it.

**Files:**
- Test: `backend/tests/test_policy_rag.py`

**Interfaces:**
- Consumes: `policy_to_chunks(policy: PolicyDocument, source: str) -> list[dict]` from `policy_rag.py`; `Item`, `Category`, `Service`, `PolicyMetadata`, `PolicyDocument` from `policy_extraction.py`.

- [ ] **Step 1: Write the tests**

Create `backend/tests/test_policy_rag.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest tests/test_policy_rag.py -q`
Expected: PASS (4 passed). (No implementation needed — `policy_to_chunks` already exists; these lock in its contract.)

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_policy_rag.py
git commit -m "test: cover policy_to_chunks composition adapter"
```

---

### Task 3: JSON artifact writer, config, and response schemas

**Files:**
- Modify: `backend/app/artifacts.py`
- Modify: `backend/app/config.py`
- Modify: `backend/.env.example`
- Modify: `backend/app/schemas.py`
- Test: append to `backend/tests/test_artifacts.py`

**Interfaces:**
- Consumes: existing `_safe_name(source: str) -> str` in `artifacts.py`.
- Produces:
  - `dump_policy_json(policy_json: str, source: str, policy_name: str, out_dir: str | Path) -> Path`
  - `settings.policy_extraction_model: str | None` (default `None`), `settings.policy_output_dir: str` (default `"policies"`)
  - `PolicyExtractionResult`, `PolicyExtractionError`, `PolicyExtractionResponse` in `schemas.py`

- [ ] **Step 1: Write the failing test for the JSON writer**

Append to `backend/tests/test_artifacts.py`:

```python
def test_dump_policy_json_writes_named_file(tmp_path):
    from app.artifacts import dump_policy_json

    path = dump_policy_json('{"policy_name": "Career Move"}', "PPG.xlsx", "Career Move", tmp_path)
    assert path == tmp_path / "PPG.xlsx__Career_Move.json"
    assert path.read_text(encoding="utf-8") == '{"policy_name": "Career Move"}'
```

- [ ] **Step 2: Run it to verify it fails**

Run: `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest tests/test_artifacts.py::test_dump_policy_json_writes_named_file -q`
Expected: FAIL — `ImportError: cannot import name 'dump_policy_json'`.

- [ ] **Step 3: Implement `dump_policy_json`**

Append to `backend/app/artifacts.py`:

```python
def dump_policy_json(policy_json: str, source: str, policy_name: str, out_dir: str | Path) -> Path:
    """Write an extracted policy's JSON to ``<out_dir>/<source>__<policy_name>.json``
    and return the path. Both name parts are sanitized; re-extracting the same
    source+policy overwrites its artifact."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_safe_name(source)}__{_safe_name(policy_name)}.json"
    path.write_text(policy_json, encoding="utf-8")
    return path
```

- [ ] **Step 4: Run it to verify it passes**

Run: `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest tests/test_artifacts.py -q`
Expected: PASS (existing artifacts tests + the new one).

- [ ] **Step 5: Add config settings**

In `backend/app/config.py`, add inside `Settings` after the `manifest_filename` line:

```python
    # Policy extraction (dedicated /documents/extract-policy path)
    policy_extraction_model: str | None = None  # falls back to chat_model; gpt-4o recommended
    policy_output_dir: str = "policies"  # where per-policy JSON is written (relative to backend/)
```

- [ ] **Step 6: Document the settings in `.env.example`**

Append to `backend/.env.example`:

```
# Policy extraction endpoint (POST /documents/extract-policy)
# Model used for policy enumeration + extraction. Leave unset to reuse CHAT_MODEL.
# gpt-4o is recommended for higher-fidelity extraction on complex PPGs.
POLICY_EXTRACTION_MODEL=
# Directory (relative to backend/) where per-policy JSON artifacts are written.
POLICY_OUTPUT_DIR=policies
```

- [ ] **Step 7: Add the response schemas**

Append to `backend/app/schemas.py`:

```python
class PolicyExtractionResult(BaseModel):
    policy_name: str
    chunk_count: int
    item_count: int
    json_path: str | None = None


class PolicyExtractionError(BaseModel):
    policy_name: str
    error: str


class PolicyExtractionResponse(BaseModel):
    source: str
    policies: list[PolicyExtractionResult]
    errors: list[PolicyExtractionError] = []
```

- [ ] **Step 8: Run the full suite to confirm nothing regressed**

Run: `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest -q`
Expected: PASS (all prior tests + the new ones).

- [ ] **Step 9: Commit**

```bash
git add backend/app/artifacts.py backend/app/config.py backend/.env.example backend/app/schemas.py backend/tests/test_artifacts.py
git commit -m "feat: policy JSON artifact writer, config, response schemas"
```

---

### Task 4: The `/documents/extract-policy` endpoint

**Files:**
- Create: `backend/app/routes/policy.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_policy_api.py`

**Interfaces:**
- Consumes: `enumerate_policies`, `extract_policy_per_service`, `scope_to_section` (Task 1); `ingest_policy` (existing, returns `{"source","chunk_count","item_count"}`); `dump_markdown`, `dump_policy_json` (Task 3); `get_converter`, `SUPPORTED_EXTENSIONS`; `PolicyExtractionResult/Error/Response` (Task 3); `settings.policy_extraction_model`, `settings.policy_output_dir` (Task 3).
- Produces: `POST /documents/extract-policy` returning `PolicyExtractionResponse`.

- [ ] **Step 1: Write the failing route tests**

Create `backend/tests/test_policy_api.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest tests/test_policy_api.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.routes.policy'`.

- [ ] **Step 3: Create the route**

Create `backend/app/routes/policy.py`:

```python
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from app.artifacts import dump_markdown, dump_policy_json
from app.config import settings
from app.conversion import SUPPORTED_EXTENSIONS, get_converter
from app.policy_extraction import enumerate_policies, extract_policy_per_service, scope_to_section
from app.policy_rag import ingest_policy
from app.schemas import (
    PolicyExtractionError,
    PolicyExtractionResponse,
    PolicyExtractionResult,
)

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
# routes/ -> app/ -> backend/
_BACKEND_DIR = Path(__file__).resolve().parents[2]


def _resolve_dir(name: str) -> Path:
    d = Path(name)
    return d if d.is_absolute() else _BACKEND_DIR / d


@router.post(
    "/documents/extract-policy",
    response_model=PolicyExtractionResponse,
    status_code=status.HTTP_200_OK,
)
async def extract_policy_endpoint(
    request: Request,
    file: Annotated[UploadFile, File()],
    sheet: str | None = Query(default=None, description="Restrict to the '## <sheet>' section"),
    policy: str | None = Query(default=None, description="Substring filter on enumerated policy names"),
) -> PolicyExtractionResponse:
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="File too large (max 10 MB)"
        )

    try:
        markdown = await get_converter().to_markdown(data, filename)
    except asyncio.TimeoutError:
        logger.warning("Conversion timed out for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="File conversion timed out"
        )
    except Exception as exc:
        logger.exception("Failed to convert file %s", filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"Could not parse file: {exc}"
        )

    # Best-effort raw-markdown dump for inspection (never breaks the request).
    if settings.save_converted_markdown:
        try:
            dump_markdown(markdown, filename, _resolve_dir(settings.converted_output_dir))
        except Exception:
            logger.exception("Failed to save converted markdown for %s", filename)

    scoped = scope_to_section(markdown, sheet) if sheet else markdown
    model = settings.policy_extraction_model or settings.chat_model

    # Blocking OpenAI call -> run off the event loop.
    try:
        names = await asyncio.to_thread(enumerate_policies, scoped, model=model)
    except Exception as exc:
        logger.exception("Policy enumeration failed for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Policy enumeration failed: {exc}"
        )

    if policy:
        needle = policy.lower()
        names = [n for n in names if needle in n.lower()]
    if not names:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="No policy content detected"
        )

    out_dir = _resolve_dir(settings.policy_output_dir)
    results: list[PolicyExtractionResult] = []
    errors: list[PolicyExtractionError] = []

    for name in names:
        # Extraction is the failure-prone step; a per-policy failure must not sink the rest.
        try:
            doc = await asyncio.to_thread(
                extract_policy_per_service,
                scoped,
                model=model,
                extra_instructions=(
                    f"Extract ONLY the '{name}' policy/column; ignore all other policies/columns."
                ),
            )
        except Exception as exc:
            logger.exception("Extraction failed for policy %r in %s", name, filename)
            errors.append(PolicyExtractionError(policy_name=name, error=str(exc)))
            continue

        # JSON artifact is a convenience deliverable — non-fatal if the write fails.
        json_path: str | None = None
        try:
            saved = dump_policy_json(doc.model_dump_json(indent=2), filename, name, out_dir)
            json_path = (
                str(saved.relative_to(_BACKEND_DIR))
                if saved.is_relative_to(_BACKEND_DIR)
                else str(saved)
            )
        except Exception:
            logger.exception("Failed to write policy JSON for %r", name)

        # RAG ingest — a failure here is a per-policy error, not a whole-request failure.
        try:
            ingest = await ingest_policy(doc, source=f"{filename} [{name}]")
        except Exception as exc:
            logger.exception("Ingest failed for policy %r in %s", name, filename)
            errors.append(PolicyExtractionError(policy_name=name, error=f"ingest failed: {exc}"))
            continue

        results.append(
            PolicyExtractionResult(
                policy_name=name,
                chunk_count=ingest["chunk_count"],
                item_count=ingest["item_count"],
                json_path=json_path,
            )
        )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"All policy extractions failed: {[e.error for e in errors]}",
        )

    logger.info("Extracted %d policies from %s (%d errors)", len(results), filename, len(errors))
    return PolicyExtractionResponse(source=filename, policies=results, errors=errors)
```

- [ ] **Step 4: Register the router**

In `backend/app/main.py`, update the routes import (line 14) and add an `include_router` call next to the others (after line 79):

```python
from app.routes import chat, documents, health, policy
```

```python
app.include_router(policy.router, tags=["ingestion"])
```

- [ ] **Step 5: Run the route tests to verify they pass**

Run: `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest tests/test_policy_api.py -q`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add backend/app/routes/policy.py backend/app/main.py backend/tests/test_policy_api.py
git commit -m "feat: POST /documents/extract-policy endpoint"
```

---

### Task 5: Documentation

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the endpoint row to README**

In `README.md`, in the API Endpoints table (after the `/documents/upload` row), add:

```markdown
| POST | /documents/extract-policy | Extract structured PolicyDocument(s) from a policy file → JSON artifacts + RAG ingest (SSE not used). Optional `?sheet=&policy=` |
```

- [ ] **Step 2: Add a Policy-extraction section to ARCHITECTURE.md**

In `ARCHITECTURE.md`, append this section after the "Structuring layer (2026-07-20)" section:

```markdown
## Policy extraction (2026-07-21)

A second, **LLM-interpretation** ingest path that lives alongside (does not replace) the
mechanical MarkItDown→structure→chunk pipeline. Where the mechanical path is deterministic
and content-agnostic, this path *interprets* a policy document into a fixed relocation-PPG
schema. Triggered explicitly via a dedicated endpoint; regular uploads are unchanged.

**Flow** (`backend/app/routes/policy.py`, `POST /documents/extract-policy`):
```
upload → MarkItDown (reused) → [optional ?sheet= slice] → enumerate_policies() (1 LLM call)
  → [optional ?policy= filter] → for each policy:
        extract_policy_per_service() (map-reduce LLM calls, run via asyncio.to_thread)
        → dump_policy_json() to POLICY_OUTPUT_DIR/<file>__<policy>.json
        → ingest_policy()  (reuses embed_texts → upsert_chunks → record_ingestion)
  → 200 { source, policies:[{policy_name, chunk_count, item_count, json_path}], errors:[…] }
```

- **`policy_extraction.py`** — `PolicyDocument` Pydantic schema (services → categories →
  items) + three LLM entry points: `enumerate_policies()` (list distinct policies/columns),
  `extract_policy()` (one-shot), `extract_policy_per_service()` (map-reduce, higher fidelity).
  `scope_to_section()` slices the Markdown to one `## <sheet>` section for the `sheet` filter.
  Extraction is document-agnostic now: a grid workbook fans out into one PolicyDocument per
  policy *column* automatically (via enumeration), and prose PDFs/DOCX enumerate to one.
- **`policy_rag.py`** — `policy_to_chunks()` (pure) flattens a PolicyDocument into clean,
  well-labeled chunks (one metadata + one per service + one per item, each with a
  `policy > service > category > item` breadcrumb and structured metadata); `ingest_policy()`
  embeds + upserts + records the manifest via the unchanged seams. Per-policy `source` is
  `"<filename> [<policy name>]"`, so multiple policies from one workbook get disjoint
  deterministic IDs and separate manifest entries.
- **`artifacts.py`** — `dump_policy_json()` writes the per-policy JSON artifact.

Config: `POLICY_EXTRACTION_MODEL` (defaults to `CHAT_MODEL`; gpt-4o recommended) and
`POLICY_OUTPUT_DIR` (default `policies`). Errors: 415/413/422 mirror the upload route;
enumeration→0 policies is 422; per-policy extraction/ingest failures are collected in
`errors[]` (partial success) and only 422 if *every* policy fails; JSON-write failure is
non-fatal. Retrieval/chat are unchanged — extracted chunks are queried like any others.
```

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md README.md
git commit -m "docs: document the policy-extraction endpoint"
```

- [ ] **Step 4: Final full-suite run**

Run: `"C:/Users/noble/miniconda3/envs/RAG-env/python.exe" -m pytest -q`
Expected: PASS — the existing suite (52) plus the 16 new tests (Task 1: 5, Task 2: 4, Task 3: 1, Task 4: 6) = 68 passing.
