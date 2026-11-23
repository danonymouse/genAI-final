"""Download and clean the two corpora per language, streaming and capped. Built for Colab.

    python scripts/fetch_corpora.py subtitles --lang ko --max-lines 1000000   # -> corpora/ko_speech.txt
    python scripts/fetch_corpora.py wiki      --lang ko --max-lines 500000    # -> corpora/ko_formal.txt

speech: OPUS OpenSubtitles, monolingual side (https://opus.nlpl.eu, data from opensubtitles.org)
formal: Wikipedia via the HuggingFace `wikimedia/wikipedia` dataset, split into sentences

Both are streamed: only the first --max-lines usable lines are downloaded and kept, never
the whole dump. Chinese text is converted to Simplified with OpenCC on both sides, so the
contrast does not confuse script (Traditional vs Simplified) with register.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "corpora"

OPUS_API = "https://opus.nlpl.eu/opusapi/"
OPUS_CODES = {"ko": ["ko"], "zh": ["zh_CN", "zh_cn", "zh"]}
OPUS_VERSIONS = ["v2024", "v2018"]
WIKI_CONFIG = {"ko": "20231101.ko", "zh": "20231101.zh"}

_HANGUL = re.compile(r"[가-힣]")
_CJK = re.compile(r"[一-鿿]")
_TAGS = re.compile(r"<[^>]*>|\{[^}]*\}")
_WS = re.compile(r"\s+")
_SENT = {"ko": re.compile(r"(?<=[.!?])\s+"), "zh": re.compile(r"(?<=[。！？])")}
_END = {"ko": (".", "!", "?"), "zh": ("。", "！", "？")}
_MIN_LEN = {"ko": 8, "zh": 5}  # Chinese packs more meaning per character


# --------------------------------------------------------------------------- helpers
def has_script(text: str, lang: str) -> bool:
    return bool((_HANGUL if lang == "ko" else _CJK).search(text))


def to_simplified():
    """Return a Traditional->Simplified converter, or identity if OpenCC is missing."""
    try:
        import opencc  # either `opencc` (C++) or `opencc-python-reimplemented`

        try:
            cc = opencc.OpenCC("t2s")
        except Exception:
            cc = opencc.OpenCC("t2s.json")
        return cc.convert
    except ImportError:
        print("WARNING: opencc not installed; Chinese text is NOT normalised to Simplified.")
        return lambda s: s


def clean_subtitle(line: str, lang: str) -> str | None:
    line = _TAGS.sub("", line)
    line = line.strip().lstrip("-–—").strip()
    line = _WS.sub(" ", line)
    if len(line) < 2 or not has_script(line, lang):
        return None
    return line


def split_sentences(text: str, lang: str) -> list[str]:
    out = []
    for para in text.split("\n"):
        para = _WS.sub(" ", para).strip()
        if not para:
            continue
        for s in _SENT[lang].split(para):
            s = s.strip()
            # keep real sentences: ends with sentence punctuation, long enough, right script
            if len(s) >= _MIN_LEN[lang] and s.endswith(_END[lang]) and has_script(s, lang):
                out.append(s)
    return out


# --------------------------------------------------------------------------- subtitles
def opus_candidates(lang: str) -> list[str]:
    """Ask the OPUS API for the mono file first; fall back to the known URL pattern."""
    urls = []
    for code in OPUS_CODES[lang]:
        try:
            r = httpx.get(OPUS_API, params={"corpus": "OpenSubtitles", "source": code,
                                            "preprocessing": "mono", "version": "latest"}, timeout=30)
            for c in r.json().get("corpora", []):
                u = c.get("url", "")
                if u.endswith(".txt.gz"):
                    urls.append(u)
        except Exception as e:
            print(f"  OPUS API lookup for {code} failed ({type(e).__name__}); trying known URLs")
    for ver in OPUS_VERSIONS:
        for code in OPUS_CODES[lang]:
            urls.append(f"https://object.pouta.csc.fi/OPUS-OpenSubtitles/{ver}/mono/{code}.txt.gz")
    return list(dict.fromkeys(urls))


def stream_gz_lines(url: str):
    """Yield decoded lines from a remote .gz file without downloading all of it."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = b""
    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        for chunk in r.iter_raw(1 << 16):
            buf += d.decompress(chunk)
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                yield ln.decode("utf-8", errors="ignore")
    if buf:
        yield buf.decode("utf-8", errors="ignore")


def fetch_subtitles(lang: str, max_lines: int, out: Path, url: str | None = None) -> None:
    conv = to_simplified() if lang == "zh" else (lambda s: s)
    for cand in ([url] if url else opus_candidates(lang)):
        try:
            with httpx.stream("GET", cand, timeout=30, follow_redirects=True) as r:
                if r.status_code != 200:
                    print(f"  {cand} -> HTTP {r.status_code}")
                    continue
        except Exception as e:
            print(f"  {cand} -> {type(e).__name__}")
            continue
        print(f"  streaming {cand}")
        n, prev = 0, None
        with out.open("w", encoding="utf-8") as f:
            for raw in stream_gz_lines(cand):
                line = clean_subtitle(raw, lang)
                if line is None or line == prev:  # skip junk and consecutive duplicates
                    continue
                prev = line
                f.write(conv(line) + "\n")
                n += 1
                if n % 200_000 == 0:
                    print(f"    {n:,} lines", flush=True)
                if n >= max_lines:
                    break
        print(f"  wrote {out} ({n:,} lines, source {cand})")
        _write_source(out, {"source": cand, "lines": n})
        return
    raise SystemExit(f"No OpenSubtitles mono file found for {lang}. Pass --url explicitly "
                     f"(find it at https://opus.nlpl.eu/OpenSubtitles).")


# --------------------------------------------------------------------------- wikipedia
def fetch_wiki(lang: str, max_lines: int, out: Path, config: str | None = None) -> None:
    from datasets import load_dataset

    config = config or WIKI_CONFIG[lang]
    conv = to_simplified() if lang == "zh" else (lambda s: s)
    print(f"  streaming wikimedia/wikipedia {config}")
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    n, next_report = 0, 100_000
    with out.open("w", encoding="utf-8") as f:
        for article in ds:
            for s in split_sentences(conv(article["text"]), lang):
                f.write(s + "\n")
                n += 1
                if n >= max_lines:
                    break
            if n >= max_lines:
                break
            if n >= next_report:
                print(f"    {n:,} sentences", flush=True)
                next_report += 100_000
    print(f"  wrote {out} ({n:,} sentences)")
    _write_source(out, {"source": f"wikimedia/wikipedia {config}", "lines": n})


def _write_source(out: Path, info: dict) -> None:
    out.with_suffix(".source.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["subtitles", "wiki"])
    ap.add_argument("--lang", choices=["ko", "zh"], required=True)
    ap.add_argument("--max-lines", type=int, default=1_000_000)
    ap.add_argument("--out", help="output file (default corpora/{lang}_{speech|formal}.txt)")
    ap.add_argument("--url", help="subtitles: explicit .txt.gz URL, skips discovery")
    ap.add_argument("--config", help="wiki: dataset config, e.g. 20231101.ko")
    a = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    name = "speech" if a.kind == "subtitles" else "formal"
    out = Path(a.out) if a.out else OUT_DIR / f"{a.lang}_{name}.txt"
    print(f"[{a.kind}:{a.lang}] up to {a.max_lines:,} lines -> {out}")
    if a.kind == "subtitles":
        fetch_subtitles(a.lang, a.max_lines, out, a.url)
    else:
        fetch_wiki(a.lang, a.max_lines, out, a.config)


if __name__ == "__main__":
    sys.exit(main())
