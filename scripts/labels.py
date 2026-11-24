"""Labelling helpers: make a spreadsheet template for a clip, and convert it to gold JSONL (D6).

    python scripts/labels.py template data/clips/ko_01.txt    # -> data/gold/ko_01.csv
    python scripts/labels.py convert  data/gold/ko_01.csv     # -> data/gold/ko_01.jsonl (+ checks)

The CSV has one row per transcript line: clip_id, line_id, text, surface, type, note.
  * A line with nothing hard in it: leave `surface` empty (that is how false positives are caught).
  * One hard span: type the phrase exactly as it appears in `text` into `surface`, pick a `type`.
  * Several spans on one line: copy the row, one row per span (same line_id).
  * Same phrase twice on a line: write `surface#2` for the second occurrence.
Offsets are computed here, so the labeller never counts characters.

Types (D5): slang, idiom, discourse_marker, register, non_literal, wordplay.
The template is deliberately NOT pre-filled with system output, so labels aren't biased by it.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold"
TYPES = {"slang", "idiom", "discourse_marker", "register", "non_literal", "wordplay"}
COLUMNS = ["clip_id", "line_id", "text", "surface", "type", "note"]


def read_clip(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def template(clip_path: Path) -> Path:
    clip_id = clip_path.stem
    GOLD.mkdir(parents=True, exist_ok=True)
    out = GOLD / f"{clip_id}.csv"
    if out.exists():
        raise SystemExit(f"{out} already exists; not overwriting labels.")
    with out.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel shows Hangul/Hanzi
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i, text in enumerate(read_clip(clip_path)):
            w.writerow([clip_id, i, text, "", "", ""])
    print(f"wrote {out} ({len(read_clip(clip_path))} lines). Fill in surface/type, then run `convert`.")
    return out


def _locate(text: str, surface: str) -> tuple[int, int]:
    occ = 1
    if "#" in surface and surface.rsplit("#", 1)[1].isdigit():
        surface, n = surface.rsplit("#", 1)
        occ = int(n)
    pos = -1
    for _ in range(occ):
        pos = text.find(surface, pos + 1)
        if pos == -1:
            raise ValueError(f"'{surface}' (occurrence {occ}) not found in: {text}")
    return pos, pos + len(surface)


def load_gold_csv(path: Path) -> list[dict]:
    """Rows -> one record per line: {clip_id, line_id, text, spans: [{start,end,surface,type,note}]}."""
    lines: dict[int, dict] = {}
    errors = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            row = {k: (v or "").strip() for k, v in row.items() if k}
            lid = int(row["line_id"])
            rec = lines.setdefault(lid, {"clip_id": row["clip_id"], "line_id": lid, "text": row["text"], "spans": []})
            if not row.get("surface"):
                continue
            if row.get("type") not in TYPES:
                errors.append(f"row {n}: type '{row.get('type')}' is not one of {sorted(TYPES)}")
                continue
            try:
                a, b = _locate(rec["text"], row["surface"])
            except ValueError as e:
                errors.append(f"row {n}: {e}")
                continue
            rec["spans"].append({"start": a, "end": b, "surface": rec["text"][a:b],
                                 "type": row["type"], "note": row.get("note", "")})
    if errors:
        raise SystemExit("Fix these rows and run again:\n  " + "\n  ".join(errors))
    return [lines[k] for k in sorted(lines)]


def load_gold(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        return load_gold_csv(path)
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def convert(csv_path: Path) -> Path:
    recs = load_gold_csv(csv_path)
    out = csv_path.with_suffix(".jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    n_spans = sum(len(r["spans"]) for r in recs)
    clean = sum(1 for r in recs if not r["spans"])
    print(f"wrote {out}: {len(recs)} lines, {n_spans} gold spans, {clean} clean lines")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("template", "convert"):
        raise SystemExit(__doc__)
    (template if sys.argv[1] == "template" else convert)(Path(sys.argv[2]))
