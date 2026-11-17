# Built for linux/amd64 so it runs on the graders' (non-Apple-silicon) machines.
# On an Apple-silicon Mac, build with:  docker build --platform linux/amd64 -t genai-final .
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# Assets are pre-built and committed (data/assets, data/cache): nothing is downloaded
# at build time beyond pip packages, and nothing at all at run time.
COPY app/ app/
COPY data/ data/

# Load both segmenters once so jieba's dictionary cache is baked into the image.
RUN python -c "from app.detector import warmup; warmup()"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ANTHROPIC_API_KEY is optional. Without it, /analyze serves cached or lexicon explanations.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
