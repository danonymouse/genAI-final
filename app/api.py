"""FastAPI service. Run: uvicorn app.api:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .detector import warmup
from .pipeline import LANGS, Pipeline
from .schemas import AnalyzeRequest, AnalyzeResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup()
    state["pipeline"] = Pipeline()
    yield
    state.clear()


app = FastAPI(
    title="Pragmatic Gap Detector",
    description="Flags the parts of real Korean and Mandarin speech that a translation hides, "
                "and explains them with retrieval-grounded explanations.",
    version="0.1.0",
    lifespan=lifespan,
)


def _pipe() -> Pipeline:
    return state["pipeline"]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    p = _pipe()
    s = p.settings
    return {
        "status": "ok",
        "explain_mode": s.explain_mode,
        "live_llm": bool(s.anthropic_api_key) and s.explain_mode in ("auto", "live"),
        "model": s.claude_model,
        "languages": {
            lang: {
                "lexicon_entries": len(c.lexicon),
                "detector": c.detector.name,
                "retriever": c.retriever.name,
                "cached_explanations": len(c.explainer.cache),
            }
            for lang, c in p.langs.items()
        },
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return _pipe().analyze(req.lang, req.get_lines())


@app.get("/examples")
def list_examples():
    d = _pipe().settings.examples_dir
    return [
        {"id": f.stem, "lang": f.stem.split("_")[0]}
        for f in sorted(d.glob("*.txt"))
        if f.stem.split("_")[0] in LANGS
    ]


@app.get("/examples/{example_id}")
def get_example(example_id: str):
    d = _pipe().settings.examples_dir
    f = d / f"{example_id}.txt"
    if not example_id.replace("_", "").isalnum() or not f.exists():
        raise HTTPException(404, "example not found")
    return {"id": example_id, "lang": example_id.split("_")[0], "text": f.read_text(encoding="utf-8")}
