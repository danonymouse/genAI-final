"""Stages 1 and 2: split a line into candidate spans, then flag the unusual ones.

Two signals flag a span:
  * lexicon   - a known surface form of a lexicon entry occurs, aligned to word boundaries
  * contrast  - the n-gram is much more frequent in casual speech than in formal writing
                score = log((f_speech + 1) / (f_formal + 1)), f = count per million tokens

The contrast signal switches on automatically when `data/assets/{lang}_freq.json.gz`
exists (built by scripts/build_assets.py). Without it, only lexicon hits are flagged.

`ngram_keys` is the single definition of how an n-gram is keyed. build_assets.py imports
it, so build-time counting and run-time lookup cannot disagree.
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .schemas import FlaggedSpan, Lang, LexiconEntry

log = logging.getLogger(__name__)

MAX_N = 3  # longest candidate n-gram, in tokens

# Characters after which a span counts as sentence-initial.
_BOUNDARY = set(".?!…。？！,，、~～\"'“”‘’()[]「」『』-—:;：；")


# --------------------------------------------------------------------------- #
# Stage 1: segmentation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Token:
    form: str
    start: int
    end: int
    tag: str = ""


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


@lru_cache(maxsize=1)
def _jieba():
    import jieba

    jieba.setLogLevel(logging.WARNING)
    jieba.initialize()
    return jieba


def warmup() -> None:
    """Load both segmenters (used at Docker build time and on API startup)."""
    _kiwi().tokenize("안녕하세요")
    _jieba().lcut("你好")


def segment(text: str, lang: Lang) -> list[Token]:
    if lang == "ko":
        return [Token(t.form, t.start, t.start + t.len, t.tag) for t in _kiwi().tokenize(text)]
    if lang == "zh":
        return [Token(w, s, e) for w, s, e in _jieba().tokenize(text) if w.strip()]
    raise ValueError(f"unsupported language: {lang}")


_PUNCT_RE = re.compile(r"^[\W\d_]+$", re.UNICODE)
# Korean tags that carry content. N*: nouns, V*: verbs/adjectives, M*: adverbs/determiners,
# IC: interjections, XR: roots, SL: foreign letters. Skip n-grams of pure particles/endings.
_KO_CONTENT = ("NN", "NP", "NR", "VV", "VA", "MA", "MM", "IC", "XR", "SL")


def _is_content(tok: Token, lang: Lang) -> bool:
    if _PUNCT_RE.match(tok.form):
        return False
    if lang == "ko":
        return tok.tag.startswith(_KO_CONTENT)
    return True


def ngram_keys(tokens: list[Token], lang: Lang, max_n: int = MAX_N):
    """Yield (key, start_idx, end_idx) for every candidate n-gram.

    key is the space-joined token forms. An n-gram is a candidate if it starts on a
    content token and does not end on punctuation.
    """
    for i in range(len(tokens)):
        for n in range(1, max_n + 1):
            j = i + n
            if j > len(tokens):
                break
            window = tokens[i:j]
            # must start on a content token (not a particle/ending) and not end on punctuation
            if not _is_content(window[0], lang) or _PUNCT_RE.match(window[-1].form):
                continue
            yield " ".join(t.form for t in window), i, j


# --------------------------------------------------------------------------- #
# Stage 2: scoring
# --------------------------------------------------------------------------- #
class ContrastTable:
    """Phrase -> (speech count, formal count), plus corpus token totals."""

    def __init__(self, counts: dict[str, list[int]], speech_total: int, formal_total: int):
        self.counts = counts
        self.speech_total = max(speech_total, 1)
        self.formal_total = max(formal_total, 1)

    @classmethod
    def load(cls, path: Path) -> "ContrastTable | None":
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["counts"], d["speech_total"], d["formal_total"])

    def score(self, key: str) -> float | None:
        c = self.counts.get(key)
        if c is None:
            return None  # pruned (< min count in speech) -> no evidence either way
        f_speech = c[0] / self.speech_total * 1e6
        f_formal = c[1] / self.formal_total * 1e6
        return math.log((f_speech + 1) / (f_formal + 1))


def _is_initial(line: str, start: int) -> bool:
    k = start - 1
    while k >= 0 and line[k].isspace():
        k -= 1
    return k < 0 or line[k] in _BOUNDARY


def _boundaries(line: str, tokens: list[Token]) -> tuple[set[int], set[int]]:
    """Positions where a span may start / end: token edges plus whitespace-word edges."""
    starts = {t.start for t in tokens}
    ends = {t.end for t in tokens}
    for m in re.finditer(r"\S+", line):
        starts.add(m.start())
        ends.add(m.end())
    return starts, ends


class Detector:
    def __init__(
        self,
        lang: Lang,
        lexicon: list[LexiconEntry],
        table: ContrastTable | None = None,
        contrast_threshold: float = 2.0,
    ):
        self.lang = lang
        self.table = table
        self.contrast_threshold = contrast_threshold
        # Longest forms first so "아니 진짜" is tried before "아니".
        self._forms = sorted(
            ((form, e) for e in lexicon for form in e.forms()), key=lambda x: -len(x[0])
        )

    @property
    def name(self) -> str:
        return "lexicon+contrast" if self.table else "lexicon"

    def detect(self, line: str) -> list[FlaggedSpan]:
        tokens = segment(line, self.lang)
        starts, ends = _boundaries(line, tokens)
        found: list[FlaggedSpan] = []

        # Signal 1: lexicon surface match, aligned to word/token boundaries.
        for form, entry in self._forms:
            pos = line.find(form)
            while pos != -1:
                end = pos + len(form)
                ok = pos in starts and end in ends
                if ok and entry.position == "initial":
                    ok = _is_initial(line, pos)
                if ok:
                    found.append(
                        FlaggedSpan(
                            start=pos, end=end, surface=form, type=entry.type,
                            score=self._contrast_for_span(tokens, pos, end),
                            detected_by=["lexicon"],
                        )
                    )
                pos = line.find(form, pos + 1)

        # Signal 2: frequency contrast over candidate n-grams.
        if self.table is not None:
            for key, i, j in ngram_keys(tokens, self.lang):
                s = self.table.score(key)
                if s is not None and s >= self.contrast_threshold:
                    a, b = tokens[i].start, tokens[j - 1].end
                    found.append(
                        FlaggedSpan(start=a, end=b, surface=line[a:b], score=round(s, 3),
                                    detected_by=["contrast"])
                    )

        return _resolve_overlaps(found)

    def _contrast_for_span(self, tokens: list[Token], start: int, end: int) -> float:
        if self.table is None:
            return 1.0
        inside = [t for t in tokens if t.start >= start and t.end <= end]
        s = self.table.score(" ".join(t.form for t in inside)) if inside else None
        return round(s, 3) if s is not None else 0.0


def _resolve_overlaps(spans: list[FlaggedSpan]) -> list[FlaggedSpan]:
    """Keep the longest span among overlaps; merge `detected_by` from what it absorbs.

    Ties on length prefer lexicon hits (they carry a type), then higher score.
    """
    spans = sorted(
        spans,
        key=lambda s: (-(s.end - s.start), "lexicon" not in s.detected_by, -s.score, s.start),
    )
    kept: list[FlaggedSpan] = []
    for s in spans:
        clash = next((k for k in kept if s.start < k.end and k.start < s.end), None)
        if clash is None:
            kept.append(s.model_copy(deep=True))
        else:
            for d in s.detected_by:
                if d not in clash.detected_by:
                    clash.detected_by.append(d)
            if clash.type is None and s.type is not None and (s.start, s.end) == (clash.start, clash.end):
                clash.type = s.type
    return sorted(kept, key=lambda s: s.start)
