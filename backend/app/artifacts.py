from __future__ import annotations

import re
from pathlib import Path

# Characters allowed in a dumped filename; everything else becomes "_".
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(source: str) -> str:
    """Reduce an upload filename to a safe basename (no path traversal)."""
    base = Path(source).name or "upload"
    return _SAFE_CHARS.sub("_", base)


def dump_markdown(markdown: str, source: str, out_dir: str | Path) -> Path:
    """Write the converted Markdown to ``<out_dir>/<source>.md`` and return the path.

    The original filename (including its extension) is kept and ``.md`` is
    appended, e.g. ``report.pdf`` -> ``report.pdf.md``, so the artifact clearly
    shows which source it came from and files of different types don't collide.
    Re-ingesting the same source overwrites its artifact.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_safe_name(source)}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def dump_policy_json(policy_json: str, source: str, policy_name: str, out_dir: str | Path) -> Path:
    """Write an extracted policy's JSON to ``<out_dir>/<source>__<policy_name>.json``
    and return the path. Both name parts are sanitized; re-extracting the same
    source+policy overwrites its artifact."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_safe_name(source)}__{_safe_name(policy_name)}.json"
    path.write_text(policy_json, encoding="utf-8")
    return path
