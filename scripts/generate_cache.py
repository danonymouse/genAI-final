"""Pre-generate LLM explanations so the container needs no LLM at all (decision D4).

Run where the model is served, normally a Colab notebook with Ollama on the free GPU:

    LLM_BASE_URL=http://localhost:11434/v1 python scripts/generate_cache.py              # data/examples/*.txt
    LLM_BASE_URL=http://localhost:11434/v1 python scripts/generate_cache.py clips/ko_01.txt

Runs the full pipeline in live mode and writes data/cache/{lang}.json. The file name
prefix (ko_..., zh_...) sets the language. Commit the cache files; the Docker image
copies them in, and /analyze reads them whenever no key is set.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402


def main() -> None:
    if not os.getenv("LLM_BASE_URL"):
        raise SystemExit("Set LLM_BASE_URL first (e.g. http://localhost:11434/v1 for Ollama).")
    os.environ["EXPLAIN_MODE"] = "live"
    settings = Settings.from_env()
    files = [Path(a) for a in sys.argv[1:]] or sorted(settings.examples_dir.glob("*.txt"))
    pipe = Pipeline(settings)

    for f in files:
        lang = f.stem.split("_")[0]
        if lang not in pipe.langs:
            print(f"skip {f} (name must start with ko_ or zh_)")
            continue
        lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        r = pipe.analyze(lang, lines)  # type: ignore[arg-type]
        print(f"{f.name}: {r.meta.n_flagged} flagged, {r.meta.n_uncertain} uncertain, mode={r.meta.mode}")
        for w in r.meta.warnings:
            print("  warning:", w)

    for lang, c in pipe.langs.items():
        c.explainer.save_cache()
        print(f"wrote {c.explainer.cache_path} ({len(c.explainer.cache)} entries)")


if __name__ == "__main__":
    main()
