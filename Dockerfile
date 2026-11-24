# Built for linux/amd64 so it runs on the graders' (non-Apple-silicon) machines.
# On an Apple-silicon Mac, build with:  docker build --platform linux/amd64 -t genai-final .
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_MODEL_DIR=/srv/models/minilm

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# Embedding model: downloaded once here (its own layer, so code changes don't re-download it),
# then checked: related phrases must embed closer than unrelated ones, or the build fails.
# Nothing is downloaded at run time.
COPY app/__init__.py app/embed.py app/
RUN python -m app.embed download && python -m app.embed selftest

# Everything else is pre-built and committed (frequency tables, word lists, cache).
COPY app/ app/
COPY data/ data/

# Build the FAISS index of the lexicon now, so the first request doesn't pay for it.
RUN python -m app.retrieval

# Load both segmenters once so jieba's dictionary cache is baked into the image.
RUN python -c "from app.detector import warmup; warmup()"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# No LLM needed: /analyze serves cached or lexicon explanations. To generate live, point
# LLM_BASE_URL at any OpenAI-compatible server (e.g. Ollama), see README.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
