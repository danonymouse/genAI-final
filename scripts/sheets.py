"""Spreadsheets for the native-speaker reviewer: no code, no JSON, no character offsets.

    python scripts/sheets.py lexicon-export             # -> sheets/lexicon_review.xlsx
    python scripts/sheets.py lexicon-import FILE.xlsx   # apply OK / FIX / DELETE back to data/lexicon/*.jsonl
    python scripts/sheets.py labels-template            # -> sheets/labelling.xlsx (one tab per clip)
    python scripts/sheets.py labels-import FILE.xlsx    # -> data/clips/*.txt, split.json, data/gold/*.csv

Lexicon review: one tab per language, one row per entry. The reviewer sets `verdict`:
  OK     -> verified = true
  FIX    -> edit the cells directly; verified = true after import
  DELETE -> entry removed
  (empty)-> left unverified
New entries can be added as new rows with an empty id.

Labelling workbook: a `clips` tab (clip roles are fixed up front: 01-02 tuning, 03-04 scoring)
and one tab per clip: paste the transcript into `text`, one utterance per row, then mark spans.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LEX = ROOT / "data" / "lexicon"
OUT = ROOT / "sheets"
TYPES = ["slang", "idiom", "discourse_marker", "register", "non_literal", "wordplay"]
HEAD = Font(bold=True, color="FFFFFF")
FILL = PatternFill("solid", fgColor="1D1D1B")
WRAP = Alignment(wrap_text=True, vertical="top")

LEX_COLS = [  # (header, entry field, width)
    ("id", "id", 9), ("headword", "headword", 16), ("other forms (separate with | )", "variants", 22),
    ("type", "type", 15), ("literal meaning", "literal", 24), ("what it really means", "meaning", 42),
    ("who says it / how casual", "register", 26), ("textbook equivalent", "textbook_equivalent", 18),
    ("why translation gets it wrong", "mt_pitfall", 32), ("example sentence", "example", 30),
    ("only at sentence start? (yes/no)", "position", 12), ("source", "source", 14),
]
VERDICT = ("verdict (OK / FIX / DELETE)", 14)
NOTE = ("note for David", 24)


def _header(ws, cols):
    ws.append([c[0] for c in cols])
    for i, c in enumerate(cols, start=1):
        cell = ws.cell(row=1, column=i)
        cell.font, cell.fill, cell.alignment = HEAD, FILL, WRAP
        ws.column_dimensions[cell.column_letter].width = c[-1]
    ws.freeze_panes = "C2"
    ws.row_dimensions[1].height = 32


def _dropdown(ws, col_letter, options, rows=2000):
    dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{col_letter}2:{col_letter}{rows}")


def _instructions(wb, title, lines):
    ws = wb.active
    ws.title = title
    ws.column_dimensions["A"].width = 110
    for ln in lines:
        ws.append([ln])
        ws.cell(row=ws.max_row, column=1).alignment = WRAP
    ws.cell(row=1, column=1).font = Font(bold=True, size=14)


# --------------------------------------------------------------------------- lexicon
def lexicon_export() -> Path:
    wb = Workbook()
    _instructions(wb, "How to", [
        "Lexicon review: check each entry in your language's tab",
        "",
        "For every row, choose a verdict in the yellow column:",
        "   OK      the entry is correct as written",
        "   FIX     something is wrong: correct the cells directly, then choose FIX",
        "   DELETE  not real, not used this way, or not worth teaching",
        "Leave the verdict empty if you are unsure; it stays unverified.",
        "",
        "Most important: is 'what it really means' right? Is the example natural? Are the other forms real?",
        "Add a missing entry: new row at the bottom, leave id empty, fill the columns, verdict OK.",
        "Do NOT add phrases from the clips you will label for scoring.",
    ])
    cols = LEX_COLS + [VERDICT, NOTE]
    for lang, name in (("ko", "Korean"), ("zh", "Mandarin")):
        ws = wb.create_sheet(name)
        _header(ws, cols)
        for line in (LEX / f"{lang}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            row = []
            for _, field, _ in LEX_COLS:
                v = e.get(field, "")
                if field == "variants":
                    v = " | ".join(v)
                elif field == "position":
                    v = "yes" if v == "initial" else "no"
                row.append(v)
            row += ["OK" if e.get("verified") else "", ""]
            ws.append(row)
        for r in ws.iter_rows(min_row=2):
            for c in r:
                c.alignment = WRAP
        yellow = PatternFill("solid", fgColor="FFF3B0")
        vcol = ws.cell(row=1, column=len(LEX_COLS) + 1).column_letter
        for r in range(2, 2000):
            ws[f"{vcol}{r}"].fill = yellow
        _dropdown(ws, vcol, ["OK", "FIX", "DELETE"])
        _dropdown(ws, ws.cell(row=1, column=4).column_letter, TYPES)
    OUT.mkdir(exist_ok=True)
    path = OUT / "lexicon_review.xlsx"
    wb.save(path)
    print(f"wrote {path}")
    return path


def lexicon_import(path: Path) -> None:
    from app.lexicon import load_lexicon
    from app.schemas import LexiconEntry

    wb = load_workbook(path, data_only=True)
    for lang, name in (("ko", "Korean"), ("zh", "Mandarin")):
        if name not in wb.sheetnames:
            continue
        ws = wb[name]
        current = {e.id: e for e in load_lexicon(LEX / f"{lang}.jsonl")}
        out, counts = [], {"OK": 0, "FIX": 0, "DELETE": 0, "unreviewed": 0, "new": 0}
        next_n = max((int(i.split("_")[1]) for i in current), default=0)
        for row in ws.iter_rows(min_row=2, values_only=True):
            cells = dict(zip([c[1] for c in LEX_COLS] + ["verdict", "note"], row))
            if not cells.get("headword"):
                continue
            verdict = str(cells.get("verdict") or "").strip().upper()
            if verdict == "DELETE":
                counts["DELETE"] += 1
                continue
            eid = str(cells.get("id") or "").strip()
            if not eid:
                next_n += 1
                eid = f"{lang}_{next_n:03d}"
                counts["new"] += 1
            base = current[eid].model_dump() if eid in current else {"source": "reviewer"}
            for _, field, _ in LEX_COLS:
                if field == "id":
                    continue
                v = cells.get(field)
                v = "" if v is None else str(v).strip()
                if field == "variants":
                    base[field] = [x.strip() for x in v.split("|") if x.strip()]
                elif field == "position":
                    base[field] = "initial" if v.lower() in ("yes", "y", "initial") else "any"
                else:
                    base[field] = v
            base.update(id=eid, lang=lang, verified=verdict in ("OK", "FIX"))
            if cells.get("note"):
                base["notes"] = str(cells["note"]).strip()
            counts[verdict if verdict in ("OK", "FIX") else "unreviewed"] += 1
            out.append(LexiconEntry(**base))  # validates type etc.
        (LEX / f"{lang}.jsonl").write_text(
            "".join(json.dumps(e.model_dump(), ensure_ascii=False) + "\n" for e in out), encoding="utf-8")
        print(f"{lang}: {len(out)} entries written · {counts}")
    print("Now run: pytest -q   (checks every entry is detected in its own example, no duplicate forms)")


# --------------------------------------------------------------------------- labels
CLIP_IDS = [f"{l}_{n:02d}" for l in ("ko", "zh") for n in range(1, 5)]
LABEL_COLS = [("line", 6), ("text (one utterance per row)", 60), ("hard phrase (copy exactly)", 24),
              ("type", 16), ("note (optional)", 30)]


def labels_template() -> Path:
    wb = Workbook()
    _instructions(wb, "How to", [
        "Labelling: what would a learner misread?",
        "",
        "1. In the 'clips' tab, write the title and link of each clip. Roles are fixed already:",
        "   clips 01-02 of each language are for tuning, 03-04 are for scoring. Do not swap them later.",
        "2. In each clip tab (ko_01 ...), paste the transcript into 'text': one utterance per row, about 20 rows.",
        "3. For each row, ask: would a learner who knows textbook Korean/Chinese misread or miss something here?",
        "   - Nothing hard: leave the rest of the row empty (this matters too).",
        "   - One hard phrase: copy it EXACTLY as written into 'hard phrase', pick a type.",
        "   - Two hard phrases in one row: add a row underneath, leave 'text' empty, fill phrase + type.",
        "   - Same phrase twice in one row: write it as  phrase#2  for the second one.",
        "",
        "Types: slang · idiom · discourse_marker (아니, 那个, fillers) · register (honorific/rude/casual shifts)",
        "       · non_literal (words not meant literally) · wordplay (puns, quotes, memes)",
        "Label from your own judgment only; do not run the system on these clips first.",
        "Use spoken content (vlogs, casual talk, comedy), not captions, and nothing the lexicon was built from.",
    ])
    ws = wb.create_sheet("clips")
    _header(ws, [("clip", 8), ("role", 10), ("title", 40), ("link", 50), ("time range used", 18)])
    for cid in CLIP_IDS:
        ws.append([cid, "tuning" if cid.endswith(("01", "02")) else "scoring", "", "", ""])
    for cid in CLIP_IDS:
        ws = wb.create_sheet(cid)
        _header(ws, LABEL_COLS)
        ws.freeze_panes = "A2"
        for i in range(40):
            ws.append([i + 1, "", "", "", ""])
        _dropdown(ws, "D", TYPES)
        for r in ws.iter_rows(min_row=2):
            for c in r:
                c.alignment = WRAP
    OUT.mkdir(exist_ok=True)
    path = OUT / "labelling.xlsx"
    wb.save(path)
    print(f"wrote {path}")
    return path


def labels_import(path: Path) -> None:
    import csv

    wb = load_workbook(path, data_only=True)
    clips_dir, gold_dir = ROOT / "data" / "clips", ROOT / "data" / "gold"
    clips_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    split = {"dev": [], "test": []}
    for row in wb["clips"].iter_rows(min_row=2, values_only=True):
        if row[0]:
            split["dev" if row[1] == "tuning" else "test"].append(row[0])
    done = []
    for cid in CLIP_IDS:
        if cid not in wb.sheetnames:
            continue
        lines, spans = [], []  # spans: (line index, surface, type, note)
        for line_no, text, phrase, typ, note in wb[cid].iter_rows(min_row=2, max_col=5, values_only=True):
            text = (str(text).strip() if text else "")
            if text:
                lines.append(text)
            if phrase:
                if not lines:
                    raise SystemExit(f"{cid} row {line_no}: a phrase before any text")
                spans.append((len(lines) - 1, str(phrase).strip(), (typ or "").strip(), (note or "")))
        if not lines:
            continue
        (clips_dir / f"{cid}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (gold_dir / f"{cid}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "line_id", "text", "surface", "type", "note"])
            marked = {i for i, *_ in spans}
            for i, t in enumerate(lines):
                if i not in marked:
                    w.writerow([cid, i, t, "", "", ""])
            for i, s, ty, n in spans:
                w.writerow([cid, i, lines[i], s, ty, n])
        done.append(cid)
    (clips_dir / "split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    sys.path.insert(0, str(ROOT / "scripts"))
    from labels import convert

    for cid in done:
        convert(gold_dir / f"{cid}.csv")  # validates phrases and types, writes JSONL
    print(f"imported {len(done)} clips: {done}; split.json written")


if __name__ == "__main__":
    cmds = {"lexicon-export": lexicon_export, "labels-template": labels_template}
    if len(sys.argv) == 2 and sys.argv[1] in cmds:
        cmds[sys.argv[1]]()
    elif len(sys.argv) == 3 and sys.argv[1] == "lexicon-import":
        lexicon_import(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "labels-import":
        labels_import(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
