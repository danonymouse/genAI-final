"""Load lexicon entries (JSONL, one entry per line) per language."""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import LexiconEntry


def load_lexicon(path: Path) -> list[LexiconEntry]:
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                entries.append(LexiconEntry(**json.loads(line)))
            except Exception as e:  # surface the bad line, don't silently skip
                raise ValueError(f"{path}:{i}: {e}") from e
    ids = [e.id for e in entries]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate ids {sorted(dupes)}")
    return entries
