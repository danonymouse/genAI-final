"""Evaluation: detection precision/recall against gold labels, and grading sheets for explanations.

    python scripts/score.py detect --split dev                 # compare detector settings on tuning clips
    python scripts/score.py detect --split test --detector lexicon+novelty   # the reported numbers
    python scripts/score.py sheet  --split test                # grading sheet for the two graders
    python scripts/score.py grades reports/grading_test.csv    # summarise a filled-in sheet

Files:
  data/clips/{clip_id}.txt        one utterance per line; clip_id starts with ko_ or zh_
  data/clips/split.json           {"dev": [...], "test": [...]}, fixed BEFORE anyone reads a label
  data/gold/{clip_id}.csv|.jsonl  labels (see scripts/labels.py)

Matching: a predicted span matches a gold span on the same line when their character ranges
overlap (exact boundaries not required). Each gold span matches at most one prediction.
Precision = matched predictions / predictions; recall = matched gold / gold. Uncertain spans
count as detections: the system flagged the phrase, it just declined to explain it.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from labels import load_gold, read_clip  # noqa: E402

CLIPS = ROOT / "data" / "clips"
GOLD = ROOT / "data" / "gold"
REPORTS = ROOT / "reports"
DETECTOR_GRID = ["lexicon", "lexicon+novelty", "lexicon+contrast", "lexicon+contrast+novelty"]


# --------------------------------------------------------------------------- matching
def overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def match_line(pred: list[dict], gold: list[dict]) -> tuple[int, list[bool], list[bool]]:
    """Greedy one-to-one overlap matching; returns (#matches, pred_matched, gold_matched)."""
    pm, gm = [False] * len(pred), [False] * len(gold)
    pairs = sorted(
        ((min(p["end"], g["end"]) - max(p["start"], g["start"]), i, j)
         for i, p in enumerate(pred) for j, g in enumerate(gold) if overlaps(p, g)),
        reverse=True,
    )
    for _, i, j in pairs:
        if not pm[i] and not gm[j]:
            pm[i] = gm[j] = True
    return sum(pm), pm, gm


@dataclasses.dataclass
class Tally:
    pred: int = 0
    gold: int = 0
    matched_pred: int = 0
    matched_gold: int = 0

    def add(self, other: "Tally") -> None:
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    @property
    def precision(self) -> float:
        return self.matched_pred / self.pred if self.pred else 0.0

    @property
    def recall(self) -> float:
        return self.matched_gold / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_lines(pred_lines: list[list[dict]], gold_lines: list[list[dict]]):
    total = Tally()
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # gold type -> [matched, total]
    misses, false_pos = [], []
    for li, (pred, gold) in enumerate(zip(pred_lines, gold_lines)):
        _, pm, gm = match_line(pred, gold)
        total.add(Tally(len(pred), len(gold), sum(pm), sum(gm)))
        for g, ok in zip(gold, gm):
            by_type[g["type"]][0] += ok
            by_type[g["type"]][1] += 1
            if not ok:
                misses.append((li, g["surface"], g["type"]))
        false_pos += [(li, p["surface"]) for p, ok in zip(pred, pm) if not ok]
    return total, dict(by_type), misses, false_pos


# --------------------------------------------------------------------------- data
def load_split() -> dict:
    path = CLIPS / "split.json"
    if not path.exists():
        raise SystemExit(f"{path} missing. Fix the dev/test split before labelling (D11).")
    return json.loads(path.read_text(encoding="utf-8"))


def gold_for(clip_id: str) -> list[dict]:
    for ext in (".jsonl", ".csv"):
        p = GOLD / f"{clip_id}{ext}"
        if p.exists():
            return load_gold(p)
    raise SystemExit(f"no labels for {clip_id} in {GOLD}")


def run_pipeline(detector: str, clip_id: str, lines: list[str]):
    os.environ["DETECTOR"] = detector
    from app.config import Settings
    from app.pipeline import Pipeline

    key = (detector,)
    if key not in _PIPES:
        _PIPES[key] = Pipeline(Settings.from_env())
    return _PIPES[key].analyze(clip_id.split("_")[0], lines)  # type: ignore[arg-type]


_PIPES: dict = {}


# --------------------------------------------------------------------------- commands
def cmd_detect(a) -> None:
    split = load_split()
    clips = split[a.split]
    detectors = [a.detector] if a.detector else DETECTOR_GRID
    out = [f"# Detection: {a.split} split", "", f"Clips: {', '.join(clips)}", ""]
    out += ["| detector | lang | gold | flagged | precision | recall | F1 |", "|---|---|---|---|---|---|---|"]
    detail: dict = {}
    for det in detectors:
        per_lang: dict[str, Tally] = defaultdict(Tally)
        types: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for cid in clips:
            gold = gold_for(cid)
            lines = [g["text"] for g in gold]
            assert lines == read_clip(CLIPS / f"{cid}.txt"), f"{cid}: gold text differs from the clip file"
            resp = run_pipeline(det, cid, lines)
            pred = [[s.model_dump() for s in ln.spans] for ln in resp.lines]
            t, bt, misses, fps = score_lines(pred, [g["spans"] for g in gold])
            lang = cid.split("_")[0]
            per_lang[lang].add(t)
            for k, (m, n) in bt.items():
                types[lang][k][0] += m
                types[lang][k][1] += n
            detail.setdefault(det, {})[cid] = {"misses": misses, "false_positives": fps,
                                               "precision": t.precision, "recall": t.recall}
        for lang, t in sorted(per_lang.items()):
            out.append(f"| {det} | {lang} | {t.gold} | {t.pred} | {t.precision:.2f} | {t.recall:.2f} | {t.f1:.2f} |")
            detail[det][f"_{lang}_recall_by_type"] = {k: f"{m}/{n}" for k, (m, n) in types[lang].items()}
    out += ["", "Overlap matching; uncertain spans count as detections. Per-clip misses and false positives "
                f"are in `reports/eval_{a.split}.json`."]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"eval_{a.split}.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (REPORTS / f"eval_{a.split}.json").write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(out))


SHEET_COLS = ["clip_id", "line_id", "text", "surface", "status", "type", "literal", "function", "why_mt_misses",
              "register", "textbook_equivalent", "explanation_source",
              "correctness (native: correct/partial/incorrect)", "usefulness (non-speaker: 0/1/2)", "comment"]


def cmd_sheet(a) -> None:
    split = load_split()
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"grading_{a.split}.csv"
    n = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(SHEET_COLS)
        for cid in split[a.split]:
            lines = read_clip(CLIPS / f"{cid}.txt")
            resp = run_pipeline(a.detector or os.getenv("DETECTOR", "lexicon"), cid, lines)
            for ln in resp.lines:
                for s in ln.spans:
                    if s.status != "explained":
                        continue  # uncertain spans have no explanation to grade
                    e = s.explanation
                    w.writerow([cid, ln.line_id, ln.text, s.surface, s.status, s.type, e.literal, e.function,
                                e.why_mt_misses, e.register, e.textbook_equivalent, s.explanation_source, "", "", ""])
                    n += 1
    print(f"wrote {path}: {n} explained spans to grade. Native speaker fills correctness; "
          f"the non-speaker fills usefulness without looking at the other column.")


def cmd_grades(a) -> None:
    corr, use = Counter(), Counter()
    by_src: dict[str, Counter] = defaultdict(Counter)
    with Path(a.sheet).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            c = row[SHEET_COLS[12]].strip().lower()
            u = row[SHEET_COLS[13]].strip()
            if c:
                corr[c] += 1
                by_src[row["explanation_source"]][c] += 1
            if u:
                use[u] += 1
    n_c, n_u = sum(corr.values()), sum(use.values())
    print(f"correctness (n={n_c}): " + ", ".join(f"{k} {v} ({v / n_c:.0%})" for k, v in corr.most_common()))
    if n_u:
        mean = sum(int(k) * v for k, v in use.items()) / n_u
        print(f"usefulness (n={n_u}): mean {mean:.2f} of 2 · " + ", ".join(f"{k}: {v}" for k, v in sorted(use.items())))
    for src, c in by_src.items():
        print(f"  {src}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--split", choices=["dev", "test"], default="dev")
    d.add_argument("--detector", help=f"one setting; default compares {DETECTOR_GRID}")
    d.set_defaults(func=cmd_detect)
    s = sub.add_parser("sheet")
    s.add_argument("--split", choices=["dev", "test"], default="test")
    s.add_argument("--detector")
    s.set_defaults(func=cmd_sheet)
    g = sub.add_parser("grades")
    g.add_argument("sheet")
    g.set_defaults(func=cmd_grades)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
