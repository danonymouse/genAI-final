"""Stage 3: find the lexicon entries that explain a flagged span.

Every retriever returns matches sorted by similarity in [0, 1]. The pipeline marks a
span `uncertain` when the best similarity is below RETRIEVAL_THRESHOLD, so no
explanation is generated without something to ground it.

M1: `KeywordRetriever` - character-bigram Dice similarity against every surface form.
M2: `FaissRetriever` - cosine similarity of sentence embeddings (span + context) over
    the lexicon index built by scripts/build_assets.py. Same interface, swap in place.
"""
from __future__ import annotations

from typing import Protocol

from .schemas import FlaggedSpan, LexiconEntry, RetrievalMatch


class Retriever(Protocol):
    name: str

    def search(self, span: FlaggedSpan, context: str, k: int = 3) -> list[tuple[RetrievalMatch, LexiconEntry]]:
        ...


def _norm(s: str) -> str:
    return "".join(s.split()).lower()


def _bigrams(s: str) -> list[str]:
    return [s[i : i + 2] for i in range(len(s) - 1)] if len(s) > 1 else [s]


def dice(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ga, gb = _bigrams(a), _bigrams(b)
    pool = list(gb)
    hits = 0
    for g in ga:
        if g in pool:
            pool.remove(g)
            hits += 1
    return 2 * hits / (len(ga) + len(gb))


class KeywordRetriever:
    name = "keyword"

    def __init__(self, lexicon: list[LexiconEntry]):
        self.lexicon = lexicon

    def search(self, span, context, k=3):
        scored = []
        for e in self.lexicon:
            sim = max(dice(span.surface, f) for f in e.forms())
            if sim > 0:
                scored.append((sim, e))
        scored.sort(key=lambda x: -x[0])
        return [
            (RetrievalMatch(entry_id=e.id, headword=e.headword, type=e.type, similarity=round(s, 3)), e)
            for s, e in scored[:k]
        ]


class FaissRetriever:  # pragma: no cover - M2
    name = "faiss"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FAISS retrieval lands in M2 (scripts/build_assets.py index).")
