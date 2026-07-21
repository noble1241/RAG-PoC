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
