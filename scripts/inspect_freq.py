"""Sanity-check a frequency table before trusting the contrast signal.

    python scripts/inspect_freq.py --lang ko            # prints, and writes reports/ko_freq_report.md

Answers three questions:
  1. Is the score distribution sensible? (percentiles)
  2. Do known slang/discourse items score high, and ordinary words low?
     Lexicon forms are the positive check; a fixed list of everyday words is the negative one.
  3. What would the detector flag on the demo transcripts at a few thresholds?
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.detector import ContrastTable, Detector, ngram_keys, segment  # noqa: E402
from app.lexicon import load_lexicon  # noqa: E402
from app.vocab import LearnerVocab  # noqa: E402

# Grammatical morphemes (particles, copula, endings) as Kiwi writes them; report-only helper.
KO_GRAMMAR = {"이", "다", "어", "아", "요", "어요", "아요", "었", "았", "겠", "시", "세요", "으시", "는", "은",
              "을", "를", "가", "고", "지", "죠", "네", "니", "냐", "야", "라고", "에", "의", "도", "만", "게",
              "기", "음", "면", "서", "어서", "아서", "니까", "라", "자", "잖아", "거든", "데", "지만", "ᆫ가",
              "예요", "에요", "습니다", "군요", "나요", "더군요", "잖아요", "님", "께", "께서", "요", "세"}

# Everyday words a learner already knows: these should NOT score high.
CONTROLS = {
    "ko": ["사람", "시간", "학교", "먹다", "가다", "오늘", "친구", "집", "물", "좋다"],
    "zh": ["时间", "学校", "朋友", "今天", "吃饭", "工作", "我们", "喜欢", "知道", "东西"],
}


def key_for(text: str, lang: str, context: str = "") -> str | None:
    """Key for `text`, segmented inside `context` when it occurs there (segmentation is
    context-sensitive, so a word alone can split differently than in a sentence)."""
    host = context if context and text in context else text
    a = host.find(text)
    b = a + len(text)
    toks = segment(host, lang)  # type: ignore[arg-type]
    for key, i, j in ngram_keys(toks, lang):  # type: ignore[arg-type]
        if toks[i].start == a and toks[j - 1].end == b:
            return key
    return None


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ko", "zh"], required=True)
    ap.add_argument("--table", help="default data/assets/{lang}_freq.json.gz")
    ap.add_argument("--min-speech", type=int, default=50, help="min speech count for the top list")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--thresholds", default="1,2,3")
    ap.add_argument("--level", type=int, help="learner level for the filter (default: ko 3, zh 4)")
    a = ap.parse_args()
    level = a.level or {"ko": 3, "zh": 4}[a.lang]
    vocab = LearnerVocab.load(ROOT / "data" / "wordlists" / f"{a.lang}_learner.tsv", level)

    path = Path(a.table or ROOT / "data" / "assets" / f"{a.lang}_freq.json.gz")
    t = ContrastTable.load(path)
    if t is None:
        raise SystemExit(f"{path} not found")
    out: list[str] = []
    say = lambda s="": (print(s), out.append(s))  # noqa: E731

    scores = {k: t.score(k) for k in t.counts}
    vals = [v for v in scores.values() if v is not None]
    say(f"# Frequency table report: {a.lang}")
    say()
    say(f"- file: `{path.name}` ({path.stat().st_size / 1e6:.1f} MB), {len(t.counts):,} phrases")
    say(f"- speech tokens: {t.speech_total:,} · formal tokens: {t.formal_total:,}")
    say(f"- score percentiles: p10 {pct(vals, 10):.2f} · p50 {pct(vals, 50):.2f} · "
        f"p90 {pct(vals, 90):.2f} · p99 {pct(vals, 99):.2f}")
    say()

    say(f"## Top {a.top} by contrast (speech count >= {a.min_speech})")
    say()
    top = sorted((k for k in t.counts if t.counts[k][0] >= a.min_speech), key=lambda k: -scores[k])[: a.top]
    say("| phrase | score | speech | formal |")
    say("|---|---|---|---|")
    for k in top:
        say(f"| {k} | {scores[k]:.2f} | {t.counts[k][0]:,} | {t.counts[k][1]:,} |")
    say()

    say("## Known items vs everyday words")
    say()
    lex = load_lexicon(ROOT / "data" / "lexicon" / f"{a.lang}.jsonl")
    pos = []
    say("| lexicon form | key | score |")
    say("|---|---|---|")
    for e in lex:
        k = key_for(e.headword, a.lang, e.example)
        s = t.score(k) if k else None
        if s is not None:
            pos.append(s)
        say(f"| {e.headword} | {k or '-'} | {'pruned' if s is None else f'{s:.2f}'} |")
    neg = []
    say()
    say("| everyday word | key | score |")
    say("|---|---|---|")
    for w in CONTROLS[a.lang]:
        k = key_for(w, a.lang)
        s = t.score(k) if k else None
        if s is not None:
            neg.append(s)
        say(f"| {w} | {k or '-'} | {'pruned' if s is None else f'{s:.2f}'} |")
    say()
    if pos and neg:
        say(f"- median score: lexicon items **{statistics.median(pos):.2f}** "
            f"({len(pos)}/{len(lex)} found) vs everyday words **{statistics.median(neg):.2f}**. "
            f"The contrast signal is useful only if the first is clearly higher.")
    say()

    if vocab is not None:
        say(f"## With the learner filter ({vocab.source}, level <= {level}, {len(vocab):,} words)")
        say()

        def known(key: str) -> bool:
            # Keys store token forms but not their tags, so the detector's content test is
            # approximated: drop Korean particles/endings (forms starting with a final-consonant
            # jamo such as ᆫ/ᆸ니다, or common grammatical morphemes). The live detector uses the
            # real tags; this only makes the offline report match it closely.
            forms = key.split()
            if a.lang != "ko":
                return vocab.covers(forms)
            if "".join(forms) in vocab.words:
                return True
            grammar = lambda f: ("\u11a8" <= f[0] <= "\u11ff") or f in KO_GRAMMAR  # noqa: E731
            return (all(f in vocab.words or grammar(f) for f in forms)
                    and any(f in vocab.words for f in forms))

        pool = sorted((k for k in t.counts if t.counts[k][0] >= a.min_speech), key=lambda k: -scores[k])[:200]
        removed = sum(known(k) for k in pool)
        say(f"- of the top 200 contrast phrases, the filter removes **{removed}** as already known")
        kept_lex = sum(1 for e in lex if not vocab.covers([e.headword]))
        say(f"- everyday control words removed: **{sum(known(w) or vocab.covers([w]) for w in CONTROLS[a.lang])}"
            f"/{len(CONTROLS[a.lang])}** · lexicon headwords the list does NOT contain: **{kept_lex}/{len(lex)}**")
        say()
        say(f"Top {a.top} after the filter:")
        say()
        say("| phrase | score | speech | formal |")
        say("|---|---|---|---|")
        survivors = [k for k in sorted((k for k in t.counts if t.counts[k][0] >= a.min_speech),
                                       key=lambda k: -scores[k]) if not known(k)][: a.top]
        for k in survivors:
            say(f"| {k} | {scores[k]:.2f} | {t.counts[k][0]:,} | {t.counts[k][1]:,} |")
        say()

    say("## Demo transcripts at different thresholds (contrast-only flags" + (", learner filter on)" if vocab else ")"))
    for thr in [float(x) for x in a.thresholds.split(",")]:
        det = Detector(a.lang, [], t, contrast_threshold=thr, vocab=vocab)  # type: ignore[arg-type]
        say()
        say(f"**threshold {thr}**")
        say()
        for f in sorted((ROOT / "data" / "examples").glob(f"{a.lang}_*.txt")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                spans = det.detect(line)
                marks = ", ".join(f"{s.surface} ({s.score:.1f})" for s in spans) or "-"
                say(f"- {line}  →  {marks}")

    rep = ROOT / "reports" / f"{a.lang}_freq_report.md"
    rep.parent.mkdir(exist_ok=True)
    rep.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nwrote {rep}")


if __name__ == "__main__":
    main()
