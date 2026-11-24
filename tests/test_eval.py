import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import labels  # noqa: E402
import score  # noqa: E402


def span(a, b, t="slang", s=""):
    return {"start": a, "end": b, "surface": s, "type": t}


def test_plan_worked_example():
    """Plan, 'How we measure' step 3: 12 gold, 15 flagged, 9 on the gold list -> P 0.60, R 0.75."""
    gold = [[span(10 * i, 10 * i + 3)] for i in range(12)]
    pred = [[span(10 * i + 1, 10 * i + 4)] if i < 9 else [] for i in range(12)]  # 9 overlapping
    pred[11] = [span(200 + k, 201 + k) for k in range(6)]  # 6 flags on nothing
    t, by_type, misses, fps = score.score_lines(pred, gold)
    assert (t.gold, t.pred, t.matched_gold) == (12, 15, 9)
    assert t.precision == pytest.approx(0.60) and t.recall == pytest.approx(0.75)
    assert len(misses) == 3 and len(fps) == 6 and by_type["slang"] == [9, 12]


def test_matching_is_one_to_one():
    m, pm, gm = score.match_line([span(0, 10)], [span(0, 3), span(5, 8)])
    assert m == 1 and gm.count(True) == 1  # one wide prediction can't claim two gold spans


def test_clean_line_flags_are_false_positives():
    t, *_ = score.score_lines([[span(0, 2)]], [[]])
    assert t.precision == 0 and t.gold == 0


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(labels.COLUMNS)
        w.writerows(rows)


def test_label_csv_offsets_and_occurrences(tmp_path):
    p = tmp_path / "ko_99.csv"
    _write_csv(p, [["ko_99", 0, "대박 진짜 대박", "대박#2", "slang", ""],
                   ["ko_99", 0, "대박 진짜 대박", "진짜", "discourse_marker", ""],
                   ["ko_99", 1, "안녕하세요", "", "", ""]])
    recs = labels.load_gold_csv(p)
    assert [(s["start"], s["end"]) for s in recs[0]["spans"]] == [(6, 8), (3, 5)]
    assert recs[1]["spans"] == []


def test_label_csv_reports_bad_rows(tmp_path):
    p = tmp_path / "ko_98.csv"
    _write_csv(p, [["ko_98", 0, "대박이야", "대박", "slangg", ""], ["ko_98", 1, "헐", "없는말", "slang", ""]])
    with pytest.raises(SystemExit) as e:
        labels.load_gold_csv(p)
    assert "row 2" in str(e.value) and "row 3" in str(e.value)


def test_template_then_detect_end_to_end(tmp_path, monkeypatch):
    clips, gold, reports = tmp_path / "clips", tmp_path / "gold", tmp_path / "reports"
    clips.mkdir()
    (clips / "ko_01.txt").write_text("아니 진짜 여기 분위기 대박이야.\n안녕하세요 여러분.\n", encoding="utf-8")
    (clips / "split.json").write_text(json.dumps({"dev": ["ko_01"], "test": []}))
    for mod in (score, labels):
        monkeypatch.setattr(mod, "GOLD", gold)
    monkeypatch.setattr(score, "CLIPS", clips)
    monkeypatch.setattr(score, "REPORTS", reports)
    csv_path = labels.template(clips / "ko_01.txt")
    rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
    rows[1][3:5] = ["아니 진짜", "discourse_marker"]
    rows.append(["ko_01", "0", rows[1][2], "대박이야", "slang", ""])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)
    labels.convert(csv_path)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    class A:
        split, detector = "dev", "lexicon"
    score.cmd_detect(A)
    report = (reports / "eval_dev.md").read_text(encoding="utf-8")
    assert "| lexicon | ko | 2 | 2 | 1.00 | 1.00 | 1.00 |" in report
