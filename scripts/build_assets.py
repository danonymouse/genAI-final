"""Build the two offline assets the service reads. Run once (locally or in Colab), commit the output.

    # frequency table: speech corpus vs formal corpus  ->  data/assets/{lang}_freq.json.gz
    python scripts/build_assets.py freq --lang ko \
        --speech corpora/ko_subtitles.txt --formal corpora/ko_wiki.txt --max-lines 500000

    # vector index of the lexicon                       ->  data/assets/{lang}_index.faiss   (M2)
    python scripts/build_assets.py index --lang ko

Inputs are plain text, one sentence per line (.txt or .txt.gz). The corpus is streamed a
line at a time, capped at --max-lines, and n-grams seen fewer than --min-count times in
speech are dropped. That keeps the table to a few MB.

N-grams are keyed with app.detector.ngram_keys, the same function the detector uses at
run time, so counting and lookup cannot disagree.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.detector import ngram_keys, segment  # noqa: E402


def stream(path: Path, max_lines: int):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        n = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line
            n += 1
            if n >= max_lines:
                return


def count(path: Path, lang: str, max_lines: int, only: set[str] | None = None) -> tuple[Counter, int]:
    """Count n-gram keys in a corpus. If `only` is given, count just those keys."""
    counts: Counter = Counter()
    tokens_total = 0
    t0 = time.time()
    for i, line in enumerate(stream(path, max_lines), 1):
        toks = segment(line, lang)  # type: ignore[arg-type]
        tokens_total += len(toks)
        for key, _, _ in ngram_keys(toks, lang):  # type: ignore[arg-type]
            if only is None or key in only:
                counts[key] += 1
        if i % 50_000 == 0:
            print(f"  {path.name}: {i:,} lines, {len(counts):,} keys, {time.time() - t0:.0f}s", flush=True)
    return counts, tokens_total


def build_freq(args) -> None:
    out = Path(args.out or ROOT / "data" / "assets" / f"{args.lang}_freq.json.gz")
    print(f"[speech] {args.speech}")
    speech, speech_total = count(Path(args.speech), args.lang, args.max_lines)
    keep = {k for k, c in speech.items() if c >= args.min_count}
    print(f"  kept {len(keep):,} of {len(speech):,} keys with count >= {args.min_count}")

    print(f"[formal] {args.formal}")
    formal, formal_total = count(Path(args.formal), args.lang, args.max_lines, only=keep)

    table = {
        "lang": args.lang,
        "speech_total": speech_total,
        "formal_total": formal_total,
        "min_count": args.min_count,
        "max_lines": args.max_lines,
        "sources": {"speech": Path(args.speech).name, "formal": Path(args.formal).name},
        "counts": {k: [speech[k], formal.get(k, 0)] for k in sorted(keep)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(keep):,} phrases)")


def build_index(args) -> None:
    raise SystemExit("index: lands in M2 (sentence-transformers + faiss over data/lexicon/{lang}.jsonl).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freq", help="build the speech-vs-formal frequency table")
    f.add_argument("--lang", choices=["ko", "zh"], required=True)
    f.add_argument("--speech", required=True, help="subtitle corpus, one line per sentence")
    f.add_argument("--formal", required=True, help="formal corpus (e.g. Wikipedia), one line per sentence")
    f.add_argument("--max-lines", type=int, default=500_000)
    f.add_argument("--min-count", type=int, default=5)
    f.add_argument("--out")
    f.set_defaults(func=build_freq)

    i = sub.add_parser("index", help="build the lexicon vector index (M2)")
    i.add_argument("--lang", choices=["ko", "zh"], required=True)
    i.set_defaults(func=build_index)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
