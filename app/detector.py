"""Stages 1 and 2: split a line into candidate spans, then flag the unusual ones.

Two signals flag a span:
  * lexicon   - a known surface form of a lexicon entry occurs, aligned to word boundaries
  * contrast  - the n-gram is much more frequent in casual speech than in formal writing
                score = log((f_speech + 1) / (f_formal + 1)), f = count per million tokens

  * novelty   - a content word that is in no learner word list (full NIKL list / all HSK
                levels) and rare in a broad frequency list (wordfreq, Zipf < NOVELTY_MAX_ZIPF):
                the neologisms film subtitles never saw (중꺽마, 알잘딱깔센, 绝绝子), while
                standard words a dictionary covers (알고리즘, 数学) are skipped. Adjacent novel
                pieces merge into one span; Korean verbs/adjectives are excluded (wordfreq lists
                surface forms, not stems).

Which signals run is set by DETECTOR ("lexicon", "lexicon+contrast", "lexicon+novelty", ...).
Contrast and novelty read `data/assets/{lang}_freq.json.gz` (built by scripts/build_assets.py).

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
from .vocab import GeneralFreq, LearnerVocab

log = logging.getLogger(__name__)

MAX_N = 3  # longest candidate n-gram, in tokens
# Bump whenever segmentation or ngram_keys changes: tables built with another version are rejected.
KEY_VERSION = 2

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
# Korean tags that can start a candidate. NN*/NR: nouns and numerals, V*: verbs/adjectives,
# M*: adverbs/determiners, IC: interjections, XR: roots, SL: foreign letters, XPN: noun prefixes
# (Kiwi splits 대박 into 대/XPN + 박/NNG, and 개웃겨 starts with 개/XPN). Particles and endings
# never start one, and neither do pronouns (NP: 내, 이거): far more frequent in speech than in
# writing, but every learner already knows them.
_KO_CONTENT = ("NN", "NR", "VV", "VA", "MA", "MM", "IC", "XR", "SL", "XPN")
_JAMO_RE = re.compile(r"[\u3131-\u318e]")  # consonant abbreviations like ㄹㅇ, ㅇㅈ, ㅋㅋ (tagged SW)


def _is_content(tok: Token, lang: Lang) -> bool:
    if lang == "ko" and tok.tag == "SW":
        return bool(_JAMO_RE.search(tok.form))
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
            if n == 1 and window[0].tag == "XPN":  # a bare prefix is not a phrase
                continue
            if lang == "ko" and window[-1].tag.startswith(("J", "VCP")):  # don't end on a particle/copula
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
        if d.get("key_version") != KEY_VERSION:
            log.warning("%s was built with key_version %s but the code uses %s; ignoring it. "
                        "Rebuild with scripts/build_assets.py.", path, d.get("key_version"), KEY_VERSION)
            return None
        return cls(d["counts"], d["speech_total"], d["formal_total"])

    def formal_pm(self, key: str) -> float | None:
        """Frequency per million tokens in the formal corpus (None if the key was pruned)."""
        c = self.counts.get(key)
        return None if c is None else c[1] / self.formal_total * 1e6

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
        vocab: LearnerVocab | None = None,
        novel_vocab: LearnerVocab | None = None,
        signals: tuple[str, ...] = ("lexicon", "contrast"),
        general: GeneralFreq | None = None,
        novelty_max_zipf: float = 3.0,
    ):
        self.lang = lang
        self.table = table
        self.contrast_threshold = contrast_threshold
        self.vocab = vocab  # contrast candidates a learner already knows are dropped
        self.novel_vocab = novel_vocab  # full learner list: anything in it is not "novel"
        self.signals = set(signals)
        self.general = general
        self.novelty_max_zipf = novelty_max_zipf
        # Longest forms first so "아니 진짜" is tried before "아니".
        self._forms = sorted(
            ((form, e) for e in lexicon for form in e.forms()), key=lambda x: -len(x[0])
        )

    @property
    def name(self) -> str:
        parts = ["lexicon"]
        if self.table is not None and "contrast" in self.signals:
            parts.append("contrast" + (f"(learner L{self.vocab.max_level})" if self.vocab else ""))
        if self.novel_vocab is not None and "novelty" in self.signals:
            parts.append("novelty")
        return "+".join(parts)

    def detect(self, line: str) -> list[FlaggedSpan]:
        tokens = segment(line, self.lang)
        starts, ends = _boundaries(line, tokens)
        found: list[FlaggedSpan] = []

        # Signal 1: lexicon surface match, aligned to word/token boundaries.
        # Segmenters split unfamiliar slang unpredictably (jieba: 太内|卷, 店颜值; Kiwi: 엄친아야
        # as one token), so only short forms need both edges aligned (아니 must not match 아니요):
        # Chinese forms of 2+ characters match anywhere; Korean forms of 3+ syllables only need
        # to start on a boundary.
        for form, entry in self._forms:
            pos = line.find(form)
            while pos != -1:
                end = pos + len(form)
                n = len(form.replace(" ", ""))
                if self.lang == "zh" and n >= 2:
                    ok = True
                elif self.lang == "ko" and n >= 3:
                    ok = pos in starts
                else:
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
        if self.table is not None and "contrast" in self.signals:
            for key, i, j in ngram_keys(tokens, self.lang):
                s = self.table.score(key)
                if s is None or s < self.contrast_threshold:
                    continue
                if self.lang == "zh" and key.replace(" ", "").isascii():
                    continue  # stray English in Chinese subtitles ("you"), not a learner gap
                if self.vocab is not None and self.vocab.covers(
                    [t.form for t in tokens[i:j] if _is_content(t, self.lang)]
                ):
                    continue  # conversational, but a learner already knows it
                a, b = tokens[i].start, tokens[j - 1].end
                found.append(
                    FlaggedSpan(start=a, end=b, surface=line[a:b], score=round(s, 3),
                                detected_by=["contrast"])
                )

        # Signal 3: novelty (words no learner list has and formal writing rarely uses).
        if self.novel_vocab is not None and "novelty" in self.signals:
            found.extend(self._novel_spans(line, tokens))

        return _resolve_overlaps(found)

    def _is_novel(self, t: Token) -> bool:
        if not _is_content(t, self.lang) or t.tag == "SW" or len(t.form) < 2:
            return False  # particles, ㅋㅋ-style symbols, single syllables (too ambiguous)
        if self.lang == "ko" and t.tag.startswith(("VV", "VA")):
            return False  # stems (싸매) never match wordfreq's surface forms
        if t.form.isascii() or any(ch.isdigit() for ch in t.form):
            return False  # numbers, stray English
        if t.form in self.novel_vocab.words:  # type: ignore[union-attr]
            return False
        if self.general is not None and self.general.get(t.form) >= self.novelty_max_zipf:
            return False  # an established word (알고리즘, 数学): any dictionary covers it
        return True

    def _novel_spans(self, line: str, tokens: list[Token]) -> list[FlaggedSpan]:
        spans: list[FlaggedSpan] = []
        run: list[Token] = []

        def flush():
            if not run:
                return
            a, b = run[0].start, run[-1].end
            s = self.table.score(" ".join(t.form for t in run)) if self.table is not None else None
            spans.append(FlaggedSpan(start=a, end=b, surface=line[a:b],
                                     score=round(s, 3) if s is not None else 0.0, detected_by=["novelty"]))
            run.clear()

        for t in tokens:
            if self._is_novel(t):
                if run and run[-1].end != t.start:
                    flush()
                run.append(t)
            elif run and t.tag == "XSN" and run[-1].end == t.start:
                run.append(t)  # keep a noun suffix with its noun: 수포 + 자 -> 수포자
            else:
                flush()
        flush()
        return spans

    def _contrast_for_span(self, tokens: list[Token], start: int, end: int) -> float:
        if self.table is None:
            return 1.0
        inside = [t for t in tokens if t.start >= start and t.end <= end]
        s = self.table.score(" ".join(t.form for t in inside)) if inside else None
        return round(s, 3) if s is not None else 0.0


def _rank(s: FlaggedSpan):
    # Lexicon hits first (they carry a type), longest first, so "아니 진짜" beats "아니".
    if "lexicon" in s.detected_by:
        return (0, -(s.end - s.start), -s.score, s.start)
    # Contrast-only: the most unusual n-gram wins, and on a tie the shorter one, so a flag
    # does not swallow its neighbours ("존맛탱", not "이거 존맛탱").
    return (1, -s.score, s.end - s.start, s.start)


def _resolve_overlaps(spans: list[FlaggedSpan]) -> list[FlaggedSpan]:
    """Keep one span per overlapping group (see _rank); merge `detected_by` from what it absorbs."""
    spans = sorted(spans, key=_rank)
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
