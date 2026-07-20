"""FIRST-PASS PROTOTYPE — schema-driven extraction of a relocation Policy Program
Guide (PPG) into structured JSON, à la mock_data/ppg_output.json.

This is intentionally NOT wired into the ingestion pipeline. It is a separate
"consumer" of the converted Markdown: where chunking.py/structuring.py produce
retrieval chunks (mechanical, deterministic, no LLM), this reads the prose and
*interprets* it into a fixed domain schema via one structured-output LLM call.

Run it:
    # from backend/ , with the machine's combined CA bundle so the OpenAI call's
    # TLS verification passes (Avast interception on this box):
    SSL_CERT_FILE=<combined-ca.pem> REQUESTS_CA_BUNDLE=<combined-ca.pem> \
    python -m app.policy_extraction path/to/converted.md --out policy.json

Notes / known first-pass limits (see the module docstring in review):
- Single LLM call over the whole document. For long PPGs this can bump the
  output-token ceiling; the production version should map-reduce per service
  (extract one `services[]` entry at a time, then merge) — the heading sections
  from structuring.py are the natural unit to drive that.
- Output is non-deterministic and unverified. Real use needs schema validation
  (free here via Pydantic) PLUS human review — the reference file contains
  human-curated annotations ("Mary confirmed…", dated updates) a model can only
  approximate.
- In OpenAI strict structured-output mode every field is always emitted, so
  optional lists come back as [] rather than being omitted (the reference file
  omits them). Semantically equivalent, slightly more verbose.
"""
from __future__ import annotations

import sys
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import settings


# --------------------------------------------------------------------------- #
# Target schema — mirrors mock_data/ppg_output.json
# --------------------------------------------------------------------------- #
class Item(BaseModel):
    """One line item within a category (typically one row of a policy table,
    enriched with the interpreted rule)."""

    item_number: str = Field(description="The item/expense code, copied verbatim from the table (e.g. '51').")
    expense_item: str = Field(description="The expense name, verbatim from the table (e.g. 'Lodging').")
    rule: str = Field(description="The full rule/benefit text for this item, as written in the source.")
    approval_required: str = Field(
        description="One of exactly: 'Yes', 'No', 'Conditional', 'Not Applicable'. "
        "'Conditional' when approval depends on a threshold or situation."
    )
    approval_details: str = Field(
        default="",
        description="If approval_required is Conditional/Yes, the specifics of what needs approving; else empty string.",
    )
    exception_conditions: str = Field(
        default="", description="Any explicit exception clause for this item; empty string if none."
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Discrete constraints/limits paraphrased from the rule "
        "(caps, day limits, eligibility). Empty list if none.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Side notes, dated updates, or clarifications. Empty list if none.",
    )
    location_constraints: list[str] = Field(
        default_factory=list, description="Location-specific rules (e.g. Guam/San Francisco caps). Empty if none."
    )
    vendor_constraints: list[str] = Field(
        default_factory=list, description="Required-vendor rules for this item. Empty if none."
    )
    required_documents: list[str] = Field(
        default_factory=list, description="Documents this item requires. Empty if none."
    )


class Category(BaseModel):
    category: str = Field(description="Category name (e.g. 'House Hunting').")
    category_code: str = Field(default="", description="Category code verbatim from the source; empty if none.")
    description: str = Field(default="", description="Category description text; empty string if none.")
    items: list[Item] = Field(default_factory=list)
    vendor_constraints: list[str] = Field(default_factory=list, description="Category-level vendor rules; empty if none.")
    notes: list[str] = Field(default_factory=list, description="Category-level notes; empty if none.")


class Service(BaseModel):
    service: str = Field(description="Service/benefit name (e.g. 'Travel Expenses', 'Lease Cancellation').")
    gross_up: str = Field(
        default="",
        description="Tax gross-up applicability, verbatim if stated: 'Yes', 'No', 'Yes or No', 'Not Applicable'.",
    )
    operational_guidelines: list[str] = Field(
        default_factory=list, description="Process/booking guidelines for the whole service. Empty if none."
    )
    required_documents: list[str] = Field(
        default_factory=list, description="Service-level required documents. Empty if none."
    )
    vendor_constraints: list[str] = Field(
        default_factory=list, description="Service-level required-vendor rules. Empty if none."
    )
    categories: list[Category] = Field(default_factory=list)


class PolicyMetadata(BaseModel):
    effective_date: str = Field(default="", description="Policy effective date as written; empty if not stated.")
    policy_description: str = Field(default="", description="Who/what the policy covers.")
    policy_document_name: str = Field(default="", description="Underlying document name/version if stated.")
    benefits_period: str = Field(default="", description="Window in which benefits must be used, if stated.")
    important_notes: list[str] = Field(
        default_factory=list, description="Policy-management notes / escalation rules. Empty if none."
    )
    contact_process: str = Field(default="", description="Employee-contact process summary, if stated.")
    inactivity_rules: str = Field(default="", description="What happens on file inactivity, if stated.")


class PolicyDocument(BaseModel):
    policy_name: str = Field(description="The policy's title.")
    policy_metadata: PolicyMetadata
    services: list[Service] = Field(default_factory=list)


# Small helper schemas used by the per-service (map-reduce) extractor below.
class PolicyHeader(BaseModel):
    policy_name: str
    policy_metadata: PolicyMetadata


class ServiceNames(BaseModel):
    services: list[str] = Field(description="Distinct top-level service/benefit names, in document order.")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """\
You extract a corporate relocation Policy Program Guide (PPG) into a strict JSON \
schema. The user message is the policy, converted to Markdown (headings, prose, \
and pipe tables).

Rules:
- Populate every field of the schema from the document. Do NOT invent facts. If \
something is not present, use an empty string "" for text fields and an empty \
list [] for list fields.
- Group the document into `services` (top-level benefits, usually headings), each \
with its `categories`, each with its line `items`. Item tables give you \
`item_number` and `expense_item` — copy those verbatim; never renumber them.
- `approval_required` must be exactly one of: "Yes", "No", "Conditional", \
"Not Applicable". Use "Conditional" when approval depends on a cap or situation, \
and put the specifics in `approval_details`.
- `conditions` is a list of short, discrete constraints paraphrased from the \
rule text (dollar caps, day limits, eligibility, "must follow travel policy"). \
Split compound rules into separate list entries.
- `notes` captures dated updates and side clarifications; `location_constraints` \
captures place-specific rules (e.g. Guam, San Francisco); `vendor_constraints` \
captures required-provider rules (e.g. must book with Ave/Manilow/Bayview).
- Preserve the source wording in `rule`/`description`; summarize only in the \
derived fields (conditions/approval_details/notes).
"""


def extract_policy(
    markdown: str,
    *,
    model: str | None = None,
    client: OpenAI | None = None,
    extra_instructions: str = "",
) -> PolicyDocument:
    """Extract a `PolicyDocument` from a policy document's Markdown via one
    structured-output LLM call. Raises RuntimeError if the model refuses.

    `model` defaults to the app's configured chat model (gpt-4o-mini). Bump to a
    stronger model (e.g. "gpt-4o") for higher-fidelity extraction on complex PPGs.
    `extra_instructions` is appended to the system prompt — use it for
    document-specific guidance (e.g. "extract only the <X> column of this grid").
    """
    client = client or OpenAI(api_key=settings.openai_api_key)
    model = model or settings.chat_model

    system = _SYSTEM_PROMPT if not extra_instructions else f"{_SYSTEM_PROMPT}\n{extra_instructions}"
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": markdown},
        ],
        response_format=PolicyDocument,
        temperature=0,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"Model refused to extract: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("Model returned no parsed output (possibly truncated at the output-token limit).")
    return message.parsed


# --------------------------------------------------------------------------- #
# Per-service extraction (map-reduce) — higher fidelity than one shot
# --------------------------------------------------------------------------- #
def _parse(client: OpenAI, model: str, system: str, user: str, response_format):
    """One structured-output call → validated Pydantic object."""
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=response_format,
        temperature=0,
    )
    msg = completion.choices[0].message
    if msg.refusal:
        raise RuntimeError(f"Model refused: {msg.refusal}")
    if msg.parsed is None:
        raise RuntimeError("No parsed output (possibly truncated at the output-token limit).")
    return msg.parsed


def extract_policy_per_service(
    markdown: str,
    *,
    model: str | None = None,
    client: OpenAI | None = None,
    extra_instructions: str = "",
    progress: bool = False,
) -> PolicyDocument:
    """Map-reduce extraction: (1) pull the header/metadata, (2) list the service
    names, (3) extract each service in its own focused call, then merge. Each
    service gets the model's full attention, which yields far better coverage and
    richer per-item fields than a single whole-document call — at the cost of
    N+2 calls. Best paired with a stronger model (e.g. gpt-4o).
    """
    client = client or OpenAI(api_key=settings.openai_api_key)
    model = model or settings.chat_model
    base = _SYSTEM_PROMPT if not extra_instructions else f"{_SYSTEM_PROMPT}\n{extra_instructions}"

    header = _parse(
        client, model,
        base + "\nTASK: Extract ONLY `policy_name` and `policy_metadata`. Leave services out.",
        markdown, PolicyHeader,
    )
    names = _parse(
        client, model,
        base + "\nTASK: List ONLY the distinct top-level service/benefit names present, in order. "
               "Do not extract their contents.",
        markdown, ServiceNames,
    ).services
    if progress:
        print(f"  services found ({len(names)}): {names}", file=sys.stderr)

    services: list[Service] = []
    for name in names:
        svc = _parse(
            client, model,
            base + f"\nTASK: Extract ONLY the single service named '{name}' as a Service object. "
                   "Include all of its categories and items; ignore every other service. "
                   "Where an item cell lists multiple categories, split it into one item per category.",
            markdown, Service,
        )
        if progress:
            n_items = sum(len(c.items) for c in svc.categories)
            print(f"  - {name!r}: {len(svc.categories)} categories, {n_items} items", file=sys.stderr)
        services.append(svc)

    return PolicyDocument(
        policy_name=header.policy_name, policy_metadata=header.policy_metadata, services=services
    )


# --------------------------------------------------------------------------- #
# CLI runner: python -m app.policy_extraction <markdown_file> [--out file.json] [--model gpt-4o]
# --------------------------------------------------------------------------- #
def _main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    md_path = Path(argv[0])
    out_path: Path | None = None
    model: str | None = None
    i = 1
    while i < len(argv):
        if argv[i] == "--out" and i + 1 < len(argv):
            out_path = Path(argv[i + 1])
            i += 2
        elif argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {argv[i]}", file=sys.stderr)
            return 2

    if not md_path.is_file():
        print(f"No such file: {md_path}", file=sys.stderr)
        return 2

    markdown = md_path.read_text(encoding="utf-8")
    print(f"Extracting from {md_path} ({len(markdown)} chars) with model "
          f"{model or settings.chat_model} ...", file=sys.stderr)
    policy = extract_policy(markdown, model=model)
    payload = policy.model_dump_json(indent=2)

    if out_path:
        out_path.write_text(payload, encoding="utf-8")
        print(f"Wrote {out_path} ({len(policy.services)} services)", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
