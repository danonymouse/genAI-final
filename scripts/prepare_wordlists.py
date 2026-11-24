"""Build the learner word lists the contrast filter reads (data/wordlists/{lang}_learner.tsv).

    python scripts/prepare_wordlists.py zh                       # HSK, downloaded from GitHub
    python scripts/prepare_wordlists.py ko path/to/list.txt      # NIKL list, downloaded by hand
    python scripts/prepare_wordlists.py general                  # wordfreq ko/zh -> {lang}_general.tsv

Mandarin: HSK 2.0 + 3.0 vocabulary, github.com/drkameleon/complete-hsk-vocabulary (MIT).
          A word's level is the lowest level it appears at in any syllabus (7 = HSK 7-9).
Korean:   국립국어원 「한국어 학습용 어휘 목록」 (2003), 5,965 words, levels A/B/C -> 1/2/3.
          https://www.korean.go.kr/front_eng/down/down_02V.do?etc_seq=70  (KOGL Type 1)
          Verbs and adjectives are stored as Kiwi-style stems (가다 -> 가); 하다/되다 verbs
          also add their noun (공부하다 -> 공부), because Kiwi splits 공부/하.

Output: one `word<TAB>level` per line, lowest level kept for duplicates.
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "wordlists"
HSK_URL = "https://raw.githubusercontent.com/drkameleon/complete-hsk-vocabulary/main/complete.min.json"
KO_LEVEL = {"A": 1, "B": 2, "C": 3}


def _write(lang: str, levels: dict[str, int], header: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{lang}_learner.tsv"
    with path.open("w", encoding="utf-8") as f:
        for w in sorted(levels, key=lambda w: (levels[w], w)):
            f.write(f"{w}\t{levels[w]}\n")
    (OUT / f"{lang}_learner.SOURCE.txt").write_text(header + "\n", encoding="utf-8")
    counts = {lv: sum(1 for v in levels.values() if v == lv) for lv in sorted(set(levels.values()))}
    print(f"wrote {path}: {len(levels):,} words, per level {counts}")
    return path


def _add(levels: dict[str, int], w: str, lv: int) -> None:
    if w and (w not in levels or lv < levels[w]):
        levels[w] = lv


# --------------------------------------------------------------------------- Mandarin
def build_zh(src: str | None) -> None:
    if src:
        data = json.loads(Path(src).read_text(encoding="utf-8"))
    else:
        import httpx

        data = httpx.get(HSK_URL, timeout=60, follow_redirects=True).json()
    levels: dict[str, int] = {}
    for item in data:
        word = item.get("s") or item.get("simplified")
        codes = item.get("l") or item.get("level") or []
        nums = [int(m.group()) for c in codes if (m := re.search(r"\d+", str(c)))]
        if word and nums:
            _add(levels, word, min(nums))
    _write("zh", levels, "HSK 2.0/3.0 vocabulary, github.com/drkameleon/complete-hsk-vocabulary (MIT). "
                         "Level = lowest level across syllabi; 7 = HSK 7-9.")


# --------------------------------------------------------------------------- Korean
def _read_text(raw: bytes) -> str:
    if raw.count("�".encode("utf-8")) > 100:
        raise SystemExit("The file is full of U+FFFD replacement characters: it was re-encoded "
                         "somewhere and the Korean text is lost. Use the original download "
                         "(zip it first if it has to be uploaded).")
    # utf-16 goes last: it "succeeds" on almost any byte string and yields garbage, so a
    # decode only counts if the header word 단어 ("word") comes out readable.
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-16"):
        try:
            text = raw.decode(enc)
            if "단어" in text[:2000]:
                return text
        except UnicodeDecodeError:
            continue
    raise SystemExit("Could not decode the Korean list (tried utf-8, cp949, euc-kr, utf-16).")


def _rows_from_file(path: Path) -> list[list[str]]:
    raw = path.read_bytes()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".txt", ".xlsx", ".xls"))
                     and not n.startswith("__MACOSX")]
            if not names:
                raise SystemExit(f"No .txt/.xlsx in {path}: {z.namelist()}")
            name = sorted(names, key=lambda n: not n.lower().endswith(".txt"))[0]
            import tempfile

            inner = Path(tempfile.mkdtemp()) / ("list" + Path(name).suffix.lower())
            inner.write_bytes(z.read(name))
            print(f"using {name} from the zip")
            return _rows_from_file(inner)
    if path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd

        df = pd.read_excel(path, header=None, dtype=str).fillna("")
        return df.values.tolist()
    text = _read_text(raw)
    return [re.split(r"\t|\s{2,}", ln.strip()) for ln in text.splitlines() if ln.strip()]


_HOMONYM = re.compile(r"\d+$")  # 가다01 -> 가다
_VERBAL = {"동", "형", "보"}  # verb, adjective, auxiliary predicate


def build_ko(src: str) -> None:
    rows = _rows_from_file(Path(src))
    levels: dict[str, int] = {}
    parsed = 0
    for row in rows:
        cells = [str(c).strip() for c in row]
        grade = next((KO_LEVEL[c] for c in reversed(cells) if c in KO_LEVEL), None)
        word = next((c for c in cells if re.search(r"[가-힣]", c)), None)
        if grade is None or word is None:
            continue  # header or junk line
        parsed += 1
        word = _HOMONYM.sub("", word.replace("-", "").replace(" ", ""))
        pos = next((c for c in cells if c in {"감", "고", "관", "대", "동", "명", "보", "부", "불", "수", "의", "형"}), "")
        _add(levels, word, grade)
        if (pos in _VERBAL or not pos) and word.endswith("다") and len(word) > 1:
            stem = word[:-1]
            _add(levels, stem, grade)  # 가다 -> 가, 예쁘다 -> 예쁘
            for suffix in ("하", "되", "시키"):  # 공부하다 -> 공부
                if stem.endswith(suffix) and len(stem) > len(suffix):
                    _add(levels, stem[: -len(suffix)], grade)
    if parsed < 1000:
        raise SystemExit(f"Only parsed {parsed} rows; the file format is not what I expected. "
                         f"First rows: {rows[:5]}")
    _write("ko", levels, "국립국어원 「한국어 학습용 어휘 목록」 (2003), "
                         "https://www.korean.go.kr/front_eng/down/down_02V.do?etc_seq=70, KOGL Type 1. "
                         "Levels A/B/C -> 1/2/3; verb/adjective stems and 하다-nouns added.")


# --------------------------------------------------------------------------- general frequency
GENERAL_MIN_ZIPF = {"ko": 0.0, "zh": 2.0}  # keep the files small; novelty thresholds sit above these


def build_general() -> None:
    """Broad word frequencies (Wikipedia, subtitles, news, books, web, social media; frozen 2021)
    from the `wordfreq` package, read from its raw tables so Korean needs no MeCab."""
    import math

    from wordfreq import get_frequency_dict

    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ("ko", "zh"):
        d = get_frequency_dict(lang)
        rows = sorted(((w, round(math.log10(f * 1e9), 2)) for w, f in d.items() if f > 0), key=lambda x: -x[1])
        rows = [(w, z) for w, z in rows if z >= GENERAL_MIN_ZIPF[lang] and "\t" not in w]
        path = OUT / f"{lang}_general.tsv"
        path.write_text("".join(f"{w}\t{z}\n" for w, z in rows), encoding="utf-8")
        (OUT / f"{lang}_general.SOURCE.txt").write_text(
            "wordfreq 3.x (Robyn Speer), https://github.com/rspeer/wordfreq. Data CC BY-SA 4.0. "
            "Zipf frequency (log10 per billion words); entries frozen at 2021.\n", encoding="utf-8")
        print(f"wrote {path}: {len(rows):,} words (zipf >= {GENERAL_MIN_ZIPF[lang]})")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "general":
        return build_general()
    if len(sys.argv) < 2 or sys.argv[1] not in ("ko", "zh"):
        raise SystemExit(__doc__)
    lang, src = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    if lang == "ko" and not src:
        raise SystemExit("Korean needs the downloaded file: prepare_wordlists.py ko path/to/list.txt")
    build_zh(src) if lang == "zh" else build_ko(src)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
