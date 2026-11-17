"""Shared data contracts. Every stage imports from here; field names do not drift.

Stage 2 (detector) -> Stage 3 (retrieval) handoff is `FlaggedSpan`.
"""
from __future__ import annotations

import warnings
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# `register` is a domain term here; pydantic warns it shadows BaseModel.register. Harmless.
warnings.filterwarnings("ignore", message='Field name "register"')

Lang = Literal["ko", "zh"]

# D5: six difficulty types. `wordplay` covers bilingual/phonetic puns.
SpanType = Literal["slang", "idiom", "discourse_marker", "register", "non_literal", "wordplay"]


# --------------------------------------------------------------------------- #
# Stage 2 -> 3 contract
# --------------------------------------------------------------------------- #
class FlaggedSpan(BaseModel):
    start: int  # character position in the line (inclusive)
    end: int  # character position in the line (exclusive)
    surface: str  # the span itself, == line[start:end]
    # Provisional at stage 2. A frequency contrast says *how* unusual a span is,
    # not *what kind* of difficulty it is, so a contrast-only flag arrives with
    # type=None and stage 3 fills it from the best lexicon match.
    type: Optional[SpanType] = None
    score: float  # contrast score (smoothed log ratio); 1.0 for a lexicon hit when no table exists
    detected_by: list[Literal["lexicon", "contrast"]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #
class LexiconEntry(BaseModel):
    id: str
    lang: Lang
    headword: str
    variants: list[str]  # surface forms to match; headword is added automatically
    type: SpanType
    literal: str  # word-for-word meaning
    meaning: str  # what it actually does in speech
    register: str  # who says it to whom
    textbook_equivalent: str  # what a textbook would say instead
    mt_pitfall: str  # why machine translation gets it wrong
    position: Literal["any", "initial"] = "any"  # "initial" = only at the start of a sentence
    example: str = ""
    notes: str = ""

    def forms(self) -> list[str]:
        seen, out = set(), []
        for f in [self.headword, *self.variants]:
            if f and f not in seen:
                seen.add(f)
                out.append(f)
        return out


# --------------------------------------------------------------------------- #
# Stage 3 / 4 outputs
# --------------------------------------------------------------------------- #
class RetrievalMatch(BaseModel):
    entry_id: str
    headword: str
    type: SpanType
    similarity: float


class Explanation(BaseModel):
    literal: str
    function: str  # what it does here, in this line
    why_mt_misses: str
    register: str
    textbook_equivalent: str


class AnalyzedSpan(FlaggedSpan):
    status: Literal["explained", "uncertain"]
    matches: list[RetrievalMatch] = Field(default_factory=list)
    explanation: Optional[Explanation] = None
    explanation_source: Optional[Literal["live", "cache", "lexicon"]] = None
    uncertain_reason: Optional[str] = None


class AnalyzedLine(BaseModel):
    line_id: int
    text: str
    spans: list[AnalyzedSpan]


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
MAX_LINES = 200


class AnalyzeRequest(BaseModel):
    lang: Lang
    text: Optional[str] = Field(None, description="Raw transcript; one utterance per line.")
    lines: Optional[list[str]] = Field(None, description="Alternative to `text`: pre-split lines.")

    @model_validator(mode="after")
    def _one_input(self):
        if (self.text is None) == (self.lines is None):
            raise ValueError("Provide exactly one of `text` or `lines`.")
        n = len(self.get_lines())
        if n == 0:
            raise ValueError("Transcript is empty.")
        if n > MAX_LINES:
            raise ValueError(f"Transcript has {n} lines; the limit is {MAX_LINES}.")
        return self

    def get_lines(self) -> list[str]:
        raw = self.lines if self.lines is not None else (self.text or "").splitlines()
        return [ln.strip() for ln in raw if ln.strip()]


class Meta(BaseModel):
    mode: Literal["live", "cache", "offline"]  # how explanations were produced overall
    model: Optional[str] = None
    detector: str
    retriever: str
    retrieval_threshold: float
    n_flagged: int
    n_uncertain: int
    warnings: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    lang: Lang
    lines: list[AnalyzedLine]
    meta: Meta
