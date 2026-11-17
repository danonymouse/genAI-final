"""Runtime settings, all from environment variables so the container needs no edits."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    anthropic_api_key: str | None
    claude_model: str
    # auto: live if a key is set, else cache, else lexicon. live/cache/offline force one path.
    explain_mode: str
    retrieval_threshold: float  # below this, a span comes back uncertain
    contrast_threshold: float  # contrast score above this flags a span (only when a table exists)

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
    def examples_dir(self) -> Path:
        return self.data_dir / "examples"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", ROOT / "data")),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
            explain_mode=os.getenv("EXPLAIN_MODE", "auto").lower(),
            retrieval_threshold=float(os.getenv("RETRIEVAL_THRESHOLD", "0.6")),
            contrast_threshold=float(os.getenv("CONTRAST_THRESHOLD", "2.0")),
        )
