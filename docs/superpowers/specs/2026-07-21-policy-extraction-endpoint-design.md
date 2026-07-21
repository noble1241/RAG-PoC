# Policy Extraction Endpoint — Format-Agnostic PolicyDocument Extraction

Design spec. Turns the standalone `policy_extraction.py` / `policy_rag.py` prototype
into a real, reusable ingestion feature: a dedicated endpoint that extracts the same
structured `PolicyDocument` JSON from **any** policy-document format (xlsx grid, pdf,
docx, …), fans a multi-policy workbook out into one document per policy automatically,
writes a JSON artifact per policy, and ingests each into Chroma for RAG.

Branch: `feat/policy-extraction`
Related: `.superpowers/sdd/progress.md` (structuring layer), `ARCHITECTURE.md`

## Goal

Today policy extraction is a CLI prototype hardcoded to one workbook: the target sheet
is sliced by hand, and a caller-supplied `extra_instructions` string names which policy
column to extract. Make it work on any policy document with no hand-slicing, producing
the same `PolicyDocument` output for every format.

## Decisions (from brainstorming)

- **Scope:** all *formats* that could be a policy document (xlsx, pdf, docx, and any
  other MarkItDown-supported type), each producing the same `PolicyDocument` JSON. The
  mechanical MarkItDown→structure→chunk pipeline is unchanged and still handles
  non-policy uploads.
- **Grid handling:** a workbook whose sheet holds several policies as side-by-side
  columns is fanned out into **one `PolicyDocument` per policy column, automatically**.
  Sheet/column become *optional narrowing filters*, not required inputs.
- **Trigger:** a **dedicated endpoint** (`POST /documents/extract-policy`). The user
  explicitly chooses to extract; regular uploads keep mechanical chunking. No
  auto-detection of policy-vs-not.
- **Output:** **both** a JSON artifact per policy (inspectable, on disk) **and** a RAG
  ingest of each policy's chunks via the existing `policy_rag` path (queryable in chat).
- **Enumeration strategy (Approach 1 — LLM-driven):** an "enumerate the distinct
  policies in this document" LLM call returns policy names/column labels; each is then
  extracted in its own focused map-reduce pass. Format-agnostic (a prose PDF enumerates
  to a single policy and flows through identically) and robust to MarkItDown's messy,
  `NaN`-riddled grid tables. Chosen over structural grid parsing (brittle) and a single
  `list[PolicyDocument]` call (hits the output-token ceiling the author already flagged).

## Architecture & data flow

```
POST /documents/extract-policy   (multipart file upload; optional ?sheet=&policy=)
  1. validate extension against SUPPORTED_EXTENSIONS      → 415 if unsupported
  2. enforce 10 MB cap (MAX_UPLOAD_BYTES)                 → 413 if too large
  3. MarkItDown → Markdown via get_converter()           → 422 on convert error/timeout
     (best-effort markdown dump, reusing dump_markdown / SAVE_CONVERTED_MARKDOWN)
  4. optional `sheet` param: restrict the Markdown to the matching `## <sheet>` section
     before enumeration (case-insensitive substring; ignored if no heading matches)
     enumerate_policies(scoped_markdown)                  → ["Career Move", "Company Request", …]
        - optional `policy` param: keep only enumerated names containing it (case-insensitive)
        - zero policies                                   → 422 "No policy content detected"
  5. for each policy name:
        extract_policy_per_service(
            markdown,
            model = POLICY_EXTRACTION_MODEL or chat_model,
            extra_instructions = f"Extract ONLY the '<name>' policy/column; ignore the others.",
        )                                                 → PolicyDocument
        write JSON  → <POLICY_OUTPUT_DIR>/<filename>__<slug(name)>.json
        ingest_policy(policy, source = "<filename> [<name>]")   (embed → upsert → manifest)
        per-policy failure (refusal / truncation / LLM error) → recorded in errors[]
  6. if every policy failed                               → 422
     else                                                 → 200
        { source, policies: [ {policy_name, chunk_count, item_count, json_path} ], errors: [ … ] }
```

A single-policy prose document (pdf/docx) enumerates to one policy and takes the exact
same path — no branch for "is this a grid."

## Components

Each unit keeps one clear purpose and reuses the existing embed/upsert/manifest seams.

### `policy_extraction.py` (extend)
- **`enumerate_policies(markdown, *, model=None, client=None) -> list[str]`** — one
  structured-output call returning distinct top-level policy names / column labels in
  document order. Reuses the existing `_parse` helper. Adds a tiny schema:
  ```python
  class PolicyNames(BaseModel):
      policies: list[str] = Field(description="Distinct policy names / policy-column "
                                              "labels present, in document order.")
  ```
  System instruction: "List each distinct policy in this document. In a workbook grid
  where policies appear as side-by-side columns, list each column's policy label; in a
  single-policy prose document, return exactly one name."
- The hardcoded per-document `extra_instructions` string is **removed**. The endpoint
  now *generates* the focus instruction from each enumerated name. `extract_policy` and
  `extract_policy_per_service` are otherwise unchanged.

### `policy_rag.py` (reuse as-is)
- `policy_to_chunks()` (pure) + `ingest_policy()` (async) reused unchanged.
- **Source identity** for a fanned-out workbook is `f"{filename} [{policy_name}]"`, so
  each policy's chunks get disjoint deterministic IDs (`make_id(source, idx, hash)`) and
  its own manifest entry — no cross-policy collisions when several come from one file.

### `routes/policy.py` (new)
- `POST /documents/extract-policy` orchestrator. Mirrors the upload route's
  validation/conversion (extension, size, MarkItDown, markdown dump), then runs
  enumerate → per-policy extract → JSON write → ingest, aggregating results and
  per-policy errors. Registered in `main.py` alongside `documents`.

### `config.py` (extend)
- `policy_extraction_model: str | None = None` — extraction model override; falls back
  to `chat_model`. `.env.example` documents it and **recommends `gpt-4o`** for fidelity.
- `policy_output_dir: str = "policies"` — where per-policy JSON is written (relative to
  `backend/`, resolved the same way as `converted_output_dir`).

## Schemas / API

Request: `multipart/form-data` with a `file` field; optional query params `sheet`
(restricts extraction to the Markdown under the matching `## <sheet>` heading before
enumeration — honors "parameterize the sheet") and `policy` (case-insensitive substring
filter on the enumerated policy names — honors "column selection"). Both default to
unset = extract everything.

Response `200`:
```json
{
  "source": "PPG_IBT_IAM_sheet.xlsx",
  "policies": [
    {"policy_name": "Career Move", "chunk_count": 41, "item_count": 33,
     "json_path": "policies/PPG_IBT_IAM_sheet.xlsx__career-move.json"},
    {"policy_name": "Company Request", "chunk_count": 38, "item_count": 30,
     "json_path": "policies/PPG_IBT_IAM_sheet.xlsx__company-request.json"}
  ],
  "errors": []
}
```
A per-policy failure appears as `{"policy_name": "...", "error": "<reason>"}` in `errors`
while other policies still succeed.

## Error handling

| Condition | Result |
|---|---|
| Unsupported extension | 415 (reuse `SUPPORTED_EXTENSIONS` gate) |
| Upload > 10 MB | 413 |
| MarkItDown convert error / timeout | 422 |
| `enumerate_policies` → 0 policies | 422 "No policy content detected" |
| One policy's extraction refuses / truncates / errors | logged; entry in `errors[]`; others proceed |
| **All** policies fail | 422 (with `errors[]` populated) |
| Upstream LLM / network failure (connection, 5xx) | 502 |

JSON-artifact write failure is **non-fatal**: log a warning, set that policy's
`json_path` to `null`, keep the successful RAG ingest. (Ingest is the primary value; the
JSON is a convenience deliverable.)

## Testing (offline, mocked — RAG-env / cp1252 console, ASCII output)

- **`enumerate_policies`** — mock the OpenAI client; assert it parses the policy list;
  fixtures for a single-policy prose doc (→ 1 name) and a multi-policy grid (→ N names).
- **`policy_to_chunks`** — pure; assert breadcrumb shape (`policy > service > category >
  item`), deterministic IDs, `content_type`/metadata, and cross-policy source
  disambiguation. (Also closes the pre-existing "new modules have no tests" gap.)
- **Route** — mock converter + `enumerate_policies` + `extract_policy_per_service`;
  assert JSON files written, `ingest_policy` invoked per policy, response shape, and the
  partial-success `errors[]` path. Optional `sheet`/`policy` filter narrows enumeration.
- No live network in any test; OpenAI and Chroma mocked, matching the existing suite
  (currently 52 passing).

## Non-goals (YAGNI)

- Auto-detecting policy vs non-policy documents (we chose an explicit endpoint).
- Human-in-the-loop curation of the extracted JSON (the reference file's "Mary
  confirmed…" annotations remain a human artifact the model only approximates).
- Schema/prompt variants for non-relocation policy types — the schema stays
  relocation-PPG–specific.
- Structural grid parsing (rejected: brittle against real, messy workbooks).
- Frontend UI for the new endpoint (API only for this iteration).

## Notes / environment

- OpenAI calls on this machine need the combined CA bundle (`SSL_CERT_FILE` /
  `REQUESTS_CA_BUNDLE`) because of Avast TLS interception — see the
  `machine-ssl-avast-interception` memory. Tests are offline and need no network.
- `mock_data/ppg_output.json` is referenced by the prototype docstring as the target
  shape but is **not present** in the repo; the JSON this feature emits is that shape.
  Adding the curated reference is out of scope here.
