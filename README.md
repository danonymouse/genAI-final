# genAI-final: Pragmatic Gap Detector

APANS5560 Applied Generative AI, final project. David Lai, Yunji Park.

A learner watching a Korean vlog hears **아니 진짜** and reads "No, really" in the subtitles. The translation is accurate and still misleading: the speaker isn't disagreeing, they're opening with emphasis, closer to "honestly". This service reads a Korean or Mandarin transcript, flags the spans a translation would hide, and explains each one, grounded in a lexicon. When nothing in the lexicon fits, it says **uncertain** rather than guess.

## Run it (Docker)

```bash
docker build -t genai-final .          # Docker image names must be lowercase
docker run --rm -p 8000:8000 genai-final
# optional, live explanations from a local open model (free), e.g. Ollama on the host:
#   ollama pull qwen3.5:9b
docker run --rm -p 8000:8000 --add-host=host.docker.internal:host-gateway \
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 genai-final
```

Then open http://localhost:8000 for the web page, or http://localhost:8000/docs for the API.

No API key and no LLM are needed. The explanations were generated ahead of time by an open model (Qwen3.5-9B) on Colab's free GPU and ship inside the image as a cache. Input that was never cached gets the lexicon entry itself. Nothing is downloaded at run time.

```bash
curl -s localhost:8000/health
curl -s localhost:8000/analyze -H 'Content-Type: application/json' \
  -d '{"lang":"ko","text":"아니 진짜 여기 분위기 대박이야.\n헐, 이 케이크 좀 보세요."}'
./scripts/smoke.sh            # end-to-end check against a running server
```

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Status, lexicon sizes, which detector, retriever and explanation mode are active |
| POST | `/analyze` | `{"lang": "ko"\|"zh", "text": "..."}` (or `"lines": [...]`) → flagged spans with explanations |
| GET | `/examples`, `/examples/{id}` | Bundled demo transcripts |
| GET | `/` | Web page: highlighted transcript with expandable cards |

Each span in the response:

```json
{"start": 0, "end": 5, "surface": "아니 진짜", "type": "discourse_marker", "score": 1.0,
 "detected_by": ["lexicon"], "status": "explained",
 "matches": [{"entry_id": "ko_001", "headword": "아니 진짜", "type": "discourse_marker", "similarity": 1.0}],
 "explanation": {"literal": "...", "function": "...", "why_mt_misses": "...", "register": "...", "textbook_equivalent": "..."},
 "explanation_source": "live | cache | lexicon", "uncertain_reason": null}
```

## How it works

| Stage | Module | What it does |
|---|---|---|
| 1 Split | `app/detector.py` | Segments the line (kiwipiepy for Korean, jieba for Mandarin) into candidate n-grams with character offsets |
| 2 Score | `app/detector.py` | Flags spans that match a lexicon form, or whose **contrast score** `log((f_speech+1)/(f_formal+1))` (per million tokens) exceeds `CONTRAST_THRESHOLD` |
| 3 Retrieve | `app/retrieval.py` | Finds the lexicon entries closest to the span. Below `RETRIEVAL_THRESHOLD` → **uncertain**, no LLM call |
| 4 Explain | `app/explain.py`, `app/llm.py` | An open LLM (Qwen3.5-9B by default, any OpenAI-compatible server) explains the span from the retrieved entries as JSON, one call per line. It can also reject the entries as not fitting the context → uncertain |
| 5 Show | `app/static/index.html` | Highlights plus expandable cards |

Stage 2 hands stage 3 a `FlaggedSpan` (`app/schemas.py`). Its `type` is provisional: a contrast-only flag says *how unusual* a span is, not *what kind*, so stage 3 fills `type` from the best match.

Explanation sources, in order (`EXPLAIN_MODE=auto`): **live** LLM if `LLM_BASE_URL` is set → **cache** (`data/cache/{lang}.json`, built by `scripts/generate_cache.py`) → **lexicon** entry fields with no LLM. If the API call fails, the request falls back to the next source and says so in `meta.warnings`.

## Offline assets

The service reads two files per language, built once and committed:

- `data/assets/{lang}_freq.json.gz`: n-gram → (speech count, formal count). Built by `scripts/build_assets.py freq` from a subtitle corpus and a formal corpus. The detector switches the contrast signal on automatically when this file exists.
- `data/assets/{lang}_index.faiss`: embedding index of the lexicon (M2).

The corpus is counted, never embedded, which keeps the image small. Raw corpora live in `corpora/` (git-ignored). Build the assets on Colab with `notebooks/01_build_freq_tables.ipynb` (CPU runtime is enough). It runs `scripts/fetch_corpora.py` (streams OPUS OpenSubtitles and Wikipedia, capped), `scripts/build_assets.py freq`, and `scripts/inspect_freq.py`, which writes a sanity report to `reports/`. Commit only `data/assets/*` and `reports/`.

## Configuration

| Variable | Default | |
|---|---|---|
| `LLM_BASE_URL` | unset | Any OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1` (Ollama). Enables live explanations |
| `LLM_MODEL` | `qwen3.5:9b` | Model name as the server knows it |
| `LLM_API_KEY` | unset | Only for hosted endpoints; local servers need none |
| `LLM_TIMEOUT` | `120` | Seconds per call |
| `EXPLAIN_MODE` | `auto` | `auto`, `live`, `cache`, `offline` |
| `RETRIEVAL_THRESHOLD` | `0.6` | Below this similarity a span is uncertain |
| `DETECTOR` | `lexicon` | `lexicon+contrast` adds frequency-contrast flags, filtered by the learner word lists |
| `LEARNER_LEVEL_KO` / `LEARNER_LEVEL_ZH` | `3` / `4` | Contrast candidates known at this level are dropped |
| `CONTRAST_THRESHOLD` | `2.0` | Contrast score needed to flag a span |

## Develop

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
uvicorn app.api:app --reload
```

## Layout

```
app/        schemas.py · config.py · lexicon.py · detector.py · retrieval.py · explain.py · llm.py · pipeline.py · api.py · static/
data/       lexicon/{ko,zh}.jsonl · wordlists/ · examples/ · cache/ · assets/
scripts/    fetch_corpora.py · build_assets.py · inspect_freq.py · prepare_wordlists.py · generate_cache.py · smoke.sh
notebooks/  01_build_freq_tables.ipynb (Colab)
tests/
docs/       decision_log.md
```

## Data and attribution

Learner word lists: 국립국어원 「한국어 학습용 어휘 목록」 (2003, KOGL Type 1, https://www.korean.go.kr/front_eng/down/down_02V.do?etc_seq=70); HSK 2.0/3.0 vocabulary from github.com/drkameleon/complete-hsk-vocabulary (MIT). Built into `data/wordlists/` by `scripts/prepare_wordlists.py`.

Subtitle corpus: OPUS OpenSubtitles (http://opus.nlpl.eu/OpenSubtitles.php), from [opensubtitles.org](https://www.opensubtitles.org). The M1 lexicons (20 Korean and 12 Mandarin entries) are hand-written.

## Status

M1 is complete. Both languages run end to end through the lexicon detector, the keyword retriever and the three-source explainer (live open LLM → cache → lexicon). M2 adds the real frequency tables, the FAISS retriever with ONNX embeddings, the full lexicons and the evaluation script. Key decisions are in `docs/decision_log.md`.
