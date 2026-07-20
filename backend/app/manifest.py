from __future__ import annotations

import json
import threading
from pathlib import Path

# Guards the read-modify-write below. Tradeoff (PoC): this is a single-process
# in-memory lock, so it does NOT protect against two processes writing the same
# manifest concurrently. Acceptable for the single-process PoC; a real system
# would use a DB or file lock.
_LOCK = threading.Lock()


def record_ingestion(manifest_path: Path, source: str, entry: dict) -> None:
    """Upsert one source's entry in the JSON manifest, incrementally.

    Keyed by source filename; re-ingesting a source replaces only its entry and
    leaves the rest untouched. A corrupt/absent manifest is treated as empty so
    a bad file never blocks new ingestions.
    """
    manifest_path = Path(manifest_path)
    with _LOCK:
        data: dict = {}
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except (json.JSONDecodeError, OSError):
                data = {}
        data[source] = entry
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
