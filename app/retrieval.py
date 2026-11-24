"""Stage 3: find the lexicon entries that explain a flagged span.

Every retriever returns matches sorted by similarity in [0, 1]. The pipeline marks a span
`uncertain` when the best similarity is below RETRIEVAL_THRESHOLD, so no explanation is
generated without something to ground it.

  KeywordRetriever  character-bigram Dice similarity against every surface form.
  HybridRetriever   FAISS (IndexFlatIP over L2-normalised MiniLM vectors, i.e. cosine) over
                    every surface form, combined with Dice:
                        similarity = max(dice, calibrated cosine)
                        calibrated = clip((cos - floor) / (1 - floor), 0, 1)
                    so exact spellings still score 1.0, and paraphrase vectors catch variants
                    the spelling misses. `floor` (SEMANTIC_FLOOR, default 0.5) maps the
                    cosine that two unrelated short phrases typically reach to 0.

The index is built from the lexicon and cached in data/assets/{lang}_index.{faiss,json};
the cache is rebuilt whenever the lexicon file or the model changes (hash check).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .schemas import FlaggedSpan, LexiconEntry, RetrievalMatch

log = logging.getLogger(__name__)


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


def _match(e: LexiconEntry, sim: float) -> tuple[RetrievalMatch, LexiconEntry]:
    return RetrievalMatch(entry_id=e.id, headword=e.headword, type=e.type, similarity=round(sim, 3)), e


class KeywordRetriever:
    name = "keyword"

    def __init__(self, lexicon: list[LexiconEntry]):
        self.lexicon = lexicon

    def dice_scores(self, surface: str) -> dict[str, float]:
        return {e.id: max(dice(surface, f) for f in e.forms()) for e in self.lexicon}

    def search(self, span, context, k=3):
        scores = self.dice_scores(span.surface)
        ranked = sorted((e for e in self.lexicon if scores[e.id] > 0), key=lambda e: -scores[e.id])
        return [_match(e, scores[e.id]) for e in ranked[:k]]


class HybridRetriever(KeywordRetriever):
    name = "hybrid"

    def __init__(self, lexicon: list[LexiconEntry], embedder, cache_stem: Path | None = None,
                 lexicon_path: Path | None = None, floor: float = 0.5):
        import faiss

        super().__init__(lexicon)
        self.embedder = embedder
        self.floor = floor
        self.by_id = {e.id: e for e in lexicon}
        self.owners: list[str] = []  # row i of the index belongs to entry owners[i]
        fingerprint = _fingerprint(lexicon_path, lexicon, getattr(embedder, "name", type(embedder).__name__))
        if cache_stem is not None and self._load(cache_stem, fingerprint):
            return
        texts = []
        for e in lexicon:
            for f in e.forms():
                texts.append(f)
                self.owners.append(e.id)
        vecs = embedder.encode(texts) if texts else np.zeros((0, embedder.dim), dtype=np.float32)
        self.index = faiss.IndexFlatIP(embedder.dim)
        self.index.add(vecs)
        if cache_stem is not None:
            self._save(cache_stem, fingerprint)

    # -------------------------------------------------------------- persistence
    def _save(self, stem: Path, fingerprint: str) -> None:
        import faiss

        stem.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(stem.with_suffix(".faiss")))
        stem.with_suffix(".json").write_text(json.dumps({"fingerprint": fingerprint, "owners": self.owners}))

    def _load(self, stem: Path, fingerprint: str) -> bool:
        import faiss

        meta, idx = stem.with_suffix(".json"), stem.with_suffix(".faiss")
        if not (meta.exists() and idx.exists()):
            return False
        m = json.loads(meta.read_text())
        if m.get("fingerprint") != fingerprint:
            log.info("%s is stale (lexicon or model changed); rebuilding", idx.name)
            return False
        self.index = faiss.read_index(str(idx))
        self.owners = m["owners"]
        return True

    # -------------------------------------------------------------- search
    def calibrate(self, cos: float) -> float:
        return float(np.clip((cos - self.floor) / (1 - self.floor), 0.0, 1.0))

    def search(self, span, context, k=3):
        scores = self.dice_scores(span.surface)
        if self.index.ntotal:
            q = self.embedder.encode([span.surface])
            dist, rows = self.index.search(q, min(self.index.ntotal, 20))
            for cos, row in zip(dist[0], rows[0]):
                if row < 0:
                    continue
                eid = self.owners[row]
                scores[eid] = max(scores[eid], self.calibrate(float(cos)))
        ranked = sorted((e for e in self.lexicon if scores[e.id] > 0), key=lambda e: -scores[e.id])
        return [_match(e, scores[e.id]) for e in ranked[:k]]


def _fingerprint(path: Path | None, lexicon: list[LexiconEntry], model: str) -> str:
    h = hashlib.sha256(model.encode())
    if path is not None and path.exists():
        h.update(path.read_bytes())
    else:
        h.update(json.dumps([e.model_dump() for e in lexicon], sort_keys=True).encode())
    return h.hexdigest()[:16]


if __name__ == "__main__":  # python -m app.retrieval  -> build/refresh the cached indexes
    from .pipeline import Pipeline

    p = Pipeline()
    for lang, c in p.langs.items():
        n = getattr(getattr(c.retriever, "index", None), "ntotal", 0)
        print(f"{lang}: retriever={c.retriever.name}, {n} vectors")
