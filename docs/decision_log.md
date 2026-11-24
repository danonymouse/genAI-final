# Decision log

Decisions that shape the build. Read this before changing the stack. Newest first within each section.

## Locked

| # | Decision | Why | Date |
|---|---|---|---|
| L1 | **The LLM is a free open model, run on Colab's free GPU.** Default Qwen3.5-9B served by Ollama. The code talks to any OpenAI-compatible endpoint (`app/llm.py`), with no paid API and no vendor SDK. | Zero cost. The model never runs inside the Docker image or on the graders' machine. | Nov 17 |
| L2 | **The graders' container serves cached explanations.** `scripts/generate_cache.py` runs on Colab against the model and writes `data/cache/{lang}.json`. The cache is committed and copied into the image. Live generation is optional via `LLM_BASE_URL`. Input that was never cached falls back to the lexicon entry. | Satisfies "Docker runs on the instructor's machine" with no key, no GPU, no run-time download (D4). | Nov 17 |
| L3 | **Embeddings use an ONNX build of the embedding model, not PyTorch.** Query spans are embedded at run time with onnxruntime. No torch or sentence-transformers in the runtime image. | PyTorch would add roughly 1 GB to the image; ONNX keeps it to a few hundred MB at most. That matters for the TAs' download and for limited laptop disk. | Nov 17 |
| L4 | **Heavy assets are built off-laptop and committed; Docker only copies them.** Corpora are downloaded and counted on Colab. Only `data/assets/*` (a few MB per language, plus the index) comes back to the repo. The Docker build downloads nothing but pip packages. | A build that downloads the OPUS and Wikipedia corpora would be slow and fragile on a grader's machine. Laptop disk is limited. | Nov 17 |
| L5 | `FlaggedSpan.type` is optional at stage 2 and filled from the best lexicon match at stage 3. | A frequency contrast measures how unusual a span is, not what kind of difficulty it is. | Nov 17 |
| L6 | Project and repo name `genAI-final`. The Docker image is tagged `genai-final`, because Docker tags must be lowercase. | | Nov 17 |
| L7 | Base image `python:3.11-slim`, built for `linux/amd64`. | The graders' machine is probably not Apple silicon. | Nov 13 |
| L8 | Segmenters: kiwipiepy (Korean) and jieba (Mandarin). Both install with pip and need no system packages. | konlpy/MeCab break Docker builds. HanLP pulls in a deep learning framework. | Nov 17 |
| L9 | Six span types: slang, idiom, discourse_marker, register, non_literal, wordplay (D5). | | Nov 13 |
| L13 | **The contrast signal is off by default (`DETECTOR=lexicon`), and when on it is gated by a learner word list.** A contrast candidate is dropped if the learner already knows it: Korean, NIKL 학습용 어휘 목록 (all 3 levels, 7,336 entries with stems); Mandarin, HSK level ≤ 4 (3,460 words). Real tables: the filter removes 194/200 (ko) and 172/200 (zh) of the top contrast phrases, and 10/10 everyday control words in both languages. Korean survivors include genuine gaps (honorifics 드세요/주무세요, 맙소사, 젠장, 설마, 어딨어); Mandarin survivors are still mostly basic phrases (你好, 我会) plus a few finds (干嘛, 伙计). The tuning clips decide whether to turn it on per language; meanwhile, survivors are used to propose lexicon entries. | Without the filter, lexicon items scored *below* everyday words in Korean (0.47 vs 1.18): the contrast measures how conversational a phrase is, not learner difficulty. Option B in the Nov 23 discussion. | Nov 23 |
| L11 | **Contrast candidates (Korean):** an n-gram must start on a content morpheme. Noun prefixes (대/XPN in 대박) and consonant abbreviations (ㄹㅇ) count; pronouns (내, 이거), particles and endings do not. It may not end on a particle or copula. Among overlapping contrast-only flags the highest score wins, shorter on ties; lexicon hits still win longest-first. Tables carry `key_version` and a mismatched table is ignored. | Found on synthetic data: 대박 was never a candidate, and flags swallowed neighbouring words. Pronouns are speech-heavy but not learning gaps. | Nov 23 |
| L12 | Chinese text on both sides is converted to Simplified with OpenCC before counting. | Wikipedia mixes Traditional and Simplified; without conversion the contrast would partly measure script, not register. | Nov 23 |
| L10 | Contrast score `log((f_speech + 1) / (f_formal + 1))`, per million tokens. N-grams are keyed by `app.detector.ngram_keys`, shared by the build and run-time code. | Smoothing avoids division by zero. One shared key function means counting and lookup cannot disagree. | Nov 17 |

## Open

| # | Decision | Proposed |
|---|---|---|
| D1 | Who owns which stages | David: pipeline, Docker, Colab. Yunji: labels, deck. Confirm. |
| D2 | Formal corpus for the contrast | Korean and Chinese Wikipedia (HF `wikimedia/wikipedia` 20231101), 500k sentences per language, on Colab. Revisit if the report shows Wikipedia skews the contrast (e.g. news would be a closer formal register). |
| D9 | Which eight clips | Four per language, about 20 lines each. |
| D10 | Lexicon per language | Korean: try 우리말샘 plus the HuggingFace slang set. Mandarin: hand-build about 150 entries. |
| D11 | Which clips tune, which score | After D9, before anyone reads a label. |
| M2 | Embedding model for the index | paraphrase-multilingual-MiniLM-L12-v2 in ONNX, quantized if retrieval quality holds. Confirm the image size in M2. |
| M2 | Exact Qwen size | qwen3.5:9b if it fits Colab's free T4 at a usable speed, otherwise qwen3.5:4b. Disable "thinking" for speed. |
