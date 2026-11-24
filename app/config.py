"""Runtime settings, all from environment variables so the container needs no edits."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    # Any OpenAI-compatible chat endpoint (Ollama, vLLM, a hosted free tier). Unset = no live LLM.
    llm_base_url: str | None
    llm_api_key: str | None  # most local servers need none
    llm_model: str
    llm_timeout: float
    # auto: live if LLM_BASE_URL is set, else cache, else lexicon. live/cache/offline force one path.
    explain_mode: str
    retrieval_threshold: float  # below this, a span comes back uncertain
    # "lexicon" or "lexicon+contrast". Contrast stays off until the tuning clips show it helps.
    detector: str
    contrast_threshold: float  # contrast score needed to flag a span
    learner_level: dict  # {"ko": 3, "zh": 4}: contrast candidates known at this level are dropped
    retriever: str  # "auto" (hybrid if the embedding model is present, else keyword), "hybrid", "keyword"
    semantic_floor: dict  # per language: cosine mapped to similarity 0 in the hybrid retriever
    @property
    def lexicon_dir(self) -> Path:
        return self.data_dir / "lexicon"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def wordlists_dir(self) -> Path:
        return self.data_dir / "wordlists"

    @property
    def examples_dir(self) -> Path:
        return self.data_dir / "examples"

    @property
    def live_enabled(self) -> bool:
        return bool(self.llm_base_url) and self.explain_mode in ("auto", "live")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", ROOT / "data")),
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            llm_api_key=os.getenv("LLM_API_KEY") or None,
            llm_model=os.getenv("LLM_MODEL", "qwen3.5:9b"),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            explain_mode=os.getenv("EXPLAIN_MODE", "auto").lower(),
            retrieval_threshold=float(os.getenv("RETRIEVAL_THRESHOLD", "0.6")),
            detector=os.getenv("DETECTOR", "lexicon").lower(),
            contrast_threshold=float(os.getenv("CONTRAST_THRESHOLD", "2.0")),
            learner_level={"ko": int(os.getenv("LEARNER_LEVEL_KO", "3")),
                           "zh": int(os.getenv("LEARNER_LEVEL_ZH", "4"))},
            retriever=os.getenv("RETRIEVER", "auto").lower(),
            semantic_floor={"ko": float(os.getenv("SEMANTIC_FLOOR_KO", "0.75")),
                            "zh": float(os.getenv("SEMANTIC_FLOOR_ZH", "0.5"))},        )
