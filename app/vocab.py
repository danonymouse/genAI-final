"""Learner vocabulary: words a learner at a given level already knows.

Used to filter contrast candidates. The frequency contrast measures how *conversational*
a phrase is, and much of conversation is vocabulary every beginner knows (안녕하세요, 吗,
你). A candidate is dropped when the learner list already covers it:

  * its content tokens joined together are a listed word (대박 if 대박 were listed), or
  * every content token is a listed word on its own (좀 + 보다, 我 + 去).

Scope, stated plainly: this filter targets unfamiliar *words*. Non-compositional phrases
built from familiar words (밥 먹었어 as a greeting, 吃瓜 as "spectating gossip") pass the
learner test as "known" and are left to the lexicon detector.

Files: data/wordlists/{lang}_learner.tsv, one `word<TAB>level` per line (built by
scripts/prepare_wordlists.py). Levels: Korean 1-3 (NIKL A-C), Mandarin 1-7 (HSK, 7 = 7-9).
"""
from __future__ import annotations

from pathlib import Path


class LearnerVocab:
    def __init__(self, words: set[str], max_level: int, source: str = ""):
        self.words = words
        self.max_level = max_level
        self.source = source

    @classmethod
    def load(cls, path: Path, max_level: int) -> "LearnerVocab | None":
        if not path.exists():
            return None
        words = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) <= max_level:
                    words.add(parts[0])
        return cls(words, max_level, path.name)

    def __len__(self) -> int:
        return len(self.words)

    def covers(self, forms: list[str]) -> bool:
        """True if a learner at this level already knows every part of the candidate."""
        if not forms:
            return False
        return "".join(forms) in self.words or all(f in self.words for f in forms)


class GeneralFreq:
    """Broad word frequency (Zipf scale) from data/wordlists/{lang}_general.tsv (wordfreq)."""

    def __init__(self, zipf: dict[str, float], source: str = ""):
        self.zipf = zipf
        self.source = source

    @classmethod
    def load(cls, path: Path) -> "GeneralFreq | None":
        if not path.exists():
            return None
        z = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                w, _, v = line.rstrip("\n").partition("\t")
                if w and v:
                    z[w] = float(v)
        return cls(z, path.name)

    def get(self, word: str) -> float:
        return self.zipf.get(word, 0.0)  # absent = rarer than anything the list kept
