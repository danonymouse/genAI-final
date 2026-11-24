import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.detector import ContrastTable, Detector, ngram_keys, segment
from app.explain import parse_items
from app.lexicon import load_lexicon
from app.llm import LLMError, OpenAICompatClient, extract_json
from app.pipeline import Pipeline
from app.retrieval import dice
from app.schemas import FlaggedSpan


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    import dataclasses

    # no key: tests never hit the network
    return dataclasses.replace(Settings.from_env(), llm_base_url=None, explain_mode="auto")


@pytest.fixture(scope="module")
def ko_lex(settings):
    return load_lexicon(settings.lexicon_dir / "ko.jsonl")


# --------------------------------------------------------------------------- lexicons
def test_lexicons_load(settings):
    for lang in ("ko", "zh"):
        lex = load_lexicon(settings.lexicon_dir / f"{lang}.jsonl")
        assert len(lex) >= 50
        assert all(e.id.startswith(lang) and e.lang == lang for e in lex)


@pytest.mark.parametrize("lang", ["ko", "zh"])
def test_every_entry_is_detected_in_its_own_example(settings, lang):
    lex = load_lexicon(settings.lexicon_dir / f"{lang}.jsonl")
    d = Detector(lang, lex)
    missed = [e.headword for e in lex
              if e.example and not any(s.surface in e.forms() for s in d.detect(e.example))]
    assert missed == []


def test_no_form_belongs_to_two_entries(settings):
    for lang in ("ko", "zh"):
        seen = {}
        for e in load_lexicon(settings.lexicon_dir / f"{lang}.jsonl"):
            for f in e.forms():
                assert f not in seen, f"{f} in {seen.get(f)} and {e.id}"
                seen[f] = e.id


# --------------------------------------------------------------------------- stage 1
def test_segment_offsets_match_text():
    for lang, text in [("ko", "아니 진짜 대박이야 ㅋㅋ"), ("zh", "我去，这个绝绝子")]:
        for t in segment(text, lang):
            assert text[t.start:t.end] == t.form or lang == "ko"  # kiwi may normalise forms
            assert 0 <= t.start < t.end <= len(text)


# --------------------------------------------------------------------------- stage 2
def test_longest_lexicon_match_wins(ko_lex):
    spans = Detector("ko", ko_lex).detect("아니 진짜 여기 분위기 대박이야.")
    assert [(s.surface, s.type) for s in spans] == [("아니 진짜", "discourse_marker"), ("대박이야", "slang")]
    assert spans[0].start == 0 and spans[0].end == 5


def test_initial_only_marker_and_word_boundaries(ko_lex):
    d = Detector("ko", ko_lex)
    assert d.detect("아니요, 저는 커피 안 마셔요.") == []  # 아니요 is a real "no"
    assert all(s.surface != "아니 진짜" for s in d.detect("그건 아니 진짜야"))  # not sentence-initial
    assert d.detect("안녕하세요 여러분, 오늘은 카페에 왔어요.") == []


def test_contrast_flag_has_no_type_until_retrieval():
    toks = segment("蜘蛛侠是什么颜色的", "zh")
    key = next(k for k, i, j in ngram_keys(toks, "zh") if k == "蜘蛛侠")
    table = ContrastTable({key: [500, 1]}, speech_total=1_000_000, formal_total=1_000_000)
    spans = Detector("zh", [], table, contrast_threshold=2.0).detect("蜘蛛侠是什么颜色的")
    assert len(spans) == 1 and spans[0].surface == "蜘蛛侠"
    assert spans[0].type is None and spans[0].detected_by == ["contrast"]
    assert spans[0].score == pytest.approx(5.525, abs=0.01)  # log(501/2)


def test_korean_candidates_include_prefixed_nouns_and_jamo():
    keys = {k for k, _, _ in ngram_keys(segment("대박이야 ㄹㅇ", "ko"), "ko")}
    assert "대 박" in keys and "ㄹㅇ" in keys and "대" not in keys
    assert not any(k.startswith(("이 ", "야")) for k in keys)  # never start on a particle/ending


def test_stale_table_is_rejected(tmp_path):
    import gzip as gz
    p = tmp_path / "ko_freq.json.gz"
    with gz.open(p, "wt", encoding="utf-8") as f:
        json.dump({"key_version": -1, "counts": {}, "speech_total": 1, "formal_total": 1}, f)
    assert ContrastTable.load(p) is None


def test_learner_filter_drops_known_candidates():
    from app.vocab import LearnerVocab
    table = ContrastTable({"你": [900, 1], "绝绝子": [500, 1], "吃 瓜": [300, 1]}, 10**6, 10**6)
    vocab = LearnerVocab({"你", "吃", "瓜"}, max_level=4)
    line = "你就是来吃瓜的，这个锅底真的绝绝子"
    assert {"绝绝子", "吃", "瓜"} <= {t.form for t in segment(line, "zh")}
    flagged = {s.surface for s in Detector("zh", [], table, 2.0, vocab).detect(line)}
    unfiltered = {s.surface for s in Detector("zh", [], table, 2.0).detect(line)}
    assert "你" in unfiltered and "你" not in flagged  # known word dropped
    assert "绝绝子" in flagged  # unknown word kept
    assert "吃瓜" not in flagged  # compositional from known parts: left to the lexicon (documented scope)


def test_contrast_is_off_by_default(settings):
    p = Pipeline(settings)
    assert settings.detector == "lexicon" and all(c.detector.table is None for c in p.langs.values())


# --------------------------------------------------------------------------- stage 3
def test_dice():
    assert dice("대박이야", "대박이야") == 1.0
    assert dice("대박", "꿀잼") == 0.0
    assert 0 < dice("장난아니에요", "장난 아니에요") <= 1.0


# --------------------------------------------------------------------------- pipeline
def test_uncertain_when_nothing_retrieved(settings, tmp_path):
    p = Pipeline(settings)
    p.langs["zh"].detector.table = ContrastTable({"蜘蛛侠": [500, 1]}, 10**6, 10**6)
    p.langs["zh"].detector.signals.add("contrast")
    r = p.analyze("zh", ["蜘蛛侠是什么颜色的？"])
    span = r.lines[0].spans[0]
    assert span.surface == "蜘蛛侠" and span.status == "uncertain"
    assert span.explanation is None and "similarity" in span.uncertain_reason
    assert r.meta.n_uncertain == 1


def test_offline_lexicon_explanations(settings):
    p = Pipeline(settings)
    r = p.analyze("ko", ["아니 진짜 여기 분위기 대박이야."])
    s = r.lines[0].spans[0]
    assert s.status == "explained" and s.explanation_source in ("lexicon", "cache")
    assert s.explanation.literal and s.explanation.function


class FakeLLM:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, 0

    def complete_json(self, system, user, max_tokens=2000):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.reply


def test_llm_rejection_marks_uncertain(settings, monkeypatch):
    """If the LLM says the entries don't fit the context, the span becomes uncertain."""
    p = Pipeline(settings)
    monkeypatch.setattr(p.langs["ko"].explainer, "llm",
                        FakeLLM({"items": [{"span_index": 0, "fits_context": False}]}))
    r = p.analyze("ko", ["헐, 이 케이크 좀 보세요."])
    s = r.lines[0].spans[0]
    assert s.status == "uncertain" and s.explanation_source == "live"
    assert r.meta.mode == "live"


def test_llm_explanation_is_used_and_cached(settings, monkeypatch):
    p = Pipeline(settings)
    ex = p.langs["ko"].explainer
    item = {"span_index": 0, "fits_context": True, "literal": "L", "function": "F",
            "why_mt_misses": "W", "register": "R", "textbook_equivalent": "T"}
    monkeypatch.setattr(ex, "llm", FakeLLM({"items": [item]}))
    monkeypatch.setattr(ex, "cache", {})
    r = p.analyze("ko", ["헐, 이 케이크 좀 보세요."])
    s = r.lines[0].spans[0]
    assert s.status == "explained" and s.explanation.function == "F" and s.explanation_source == "live"
    assert len(ex.cache) == 1
    # same line again with the LLM gone -> served from cache
    monkeypatch.setattr(ex, "llm", None)
    s2 = p.analyze("ko", ["헐, 이 케이크 좀 보세요."]).lines[0].spans[0]
    assert s2.explanation_source == "cache" and s2.explanation.function == "F"


def test_llm_failure_falls_back(settings, monkeypatch):
    p = Pipeline(settings)
    monkeypatch.setattr(p.langs["ko"].explainer, "llm", FakeLLM(exc=RuntimeError("no network")))
    r = p.analyze("ko", ["헐, 이 케이크 좀 보세요."])
    assert r.lines[0].spans[0].status == "explained"
    assert r.meta.warnings and r.meta.mode in ("offline", "cache")


def test_malformed_items_fall_back_per_span():
    out = parse_items({"items": [{"span_index": 0, "function": ""}, {"span_index": "1", "function": "ok"},
                                 "junk", {"span_index": 9, "function": "x"}]}, 2)
    assert out[0] is None and out[1].explanation.function == "ok"
    assert parse_items({"nope": 1}, 2) == [None, None]


def test_extract_json_is_lenient():
    assert extract_json('{"items": []}') == {"items": []}
    assert extract_json('<think>hmm {not json}</think>\n```json\n{"items": [1]}\n```') == {"items": [1]}
    assert extract_json('Sure! Here you go: {"a": "b}"} hope that helps') == {"a": "b}"}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_openai_compatible_client_over_http():
    """Real HTTP round trip against a fake server that rejects JSON mode and 'thinks' first."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if "response_format" in body:
                self.send_response(400); self.end_headers(); self.wfile.write(b"no json mode"); return
            content = '<think>reasoning...</think>{"items": [{"span_index": 0, "function": "ok"}]}'
            out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = OpenAICompatClient(f"http://127.0.0.1:{srv.server_port}/v1", "qwen3.5:9b")
        assert c.complete_json("sys", "user") == {"items": [{"span_index": 0, "function": "ok"}]}
        assert c.complete_json("sys", "user")["items"][0]["function"] == "ok"
        # first call tried JSON mode then retried without; second call skips JSON mode
        assert ["response_format" in b for b in seen] == [True, False, False]
        assert seen[0]["model"] == "qwen3.5:9b" and seen[0]["temperature"] == 0
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(monkeypatch_module):
    from app.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    mp.delenv("LLM_BASE_URL", raising=False)
    yield mp
    mp.undo()


def test_api_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["languages"]["ko"]["lexicon_entries"] >= 50


def test_api_analyze_shape(client):
    text = client.get("/examples/ko_demo").json()["text"]
    r = client.post("/analyze", json={"lang": "ko", "text": text})
    assert r.status_code == 200
    body = r.json()
    assert len(body["lines"]) == 10
    for line in body["lines"]:
        for s in line["spans"]:
            assert line["text"][s["start"]:s["end"]] == s["surface"]
            assert {"start", "end", "surface", "type", "score", "status"} <= s.keys()


def test_api_rejects_bad_input(client):
    assert client.post("/analyze", json={"lang": "ja", "text": "x"}).status_code == 422
    assert client.post("/analyze", json={"lang": "ko"}).status_code == 422
    assert client.post("/analyze", json={"lang": "ko", "text": "   "}).status_code == 422
    assert client.get("/examples/../../etc").status_code == 404


def test_web_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Pragmatic Gap Detector" in r.text


# --------------------------------------------------------------------------- hybrid retrieval (FAISS)
class CharGramEmbedder:
    """Deterministic stand-in for MiniLM: hashed character 1-2 grams, L2-normalised."""
    dim = 384
    name = "chargram-test"

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        import numpy as np
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            t = "".join(t.split())
            for g in list(t) + [t[i:i + 2] for i in range(len(t) - 1)]:
                out[r, hash(g) % self.dim] += 1.0
        out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return out


def _span(surface):
    return FlaggedSpan(start=0, end=len(surface), surface=surface, score=1.0)


def test_hybrid_exact_match_scores_one_and_ranks_first(ko_lex):
    from app.retrieval import HybridRetriever
    r = HybridRetriever(ko_lex, CharGramEmbedder())
    top = r.search(_span("대박이야"), "", k=3)
    assert top[0][1].headword == "대박" and top[0][0].similarity == 1.0
    assert r.index.ntotal == sum(len(e.forms()) for e in ko_lex)


def test_hybrid_semantic_catches_unlisted_variant(ko_lex):
    from app.retrieval import HybridRetriever, KeywordRetriever
    span = _span("존맛탱구리")  # an unlisted, playful extension of 존맛탱
    kw = KeywordRetriever(ko_lex).search(span, "", k=1)[0][0].similarity
    hy = HybridRetriever(ko_lex, CharGramEmbedder(), floor=0.3).search(span, "", k=1)[0]
    assert hy[1].headword == "존맛" and hy[0].similarity >= kw


def test_hybrid_unrelated_stays_below_threshold(ko_lex):
    from app.retrieval import HybridRetriever
    top = HybridRetriever(ko_lex, CharGramEmbedder()).search(_span("蜘蛛侠"), "", k=1)
    assert not top or top[0][0].similarity < 0.6


def test_hybrid_index_cache_is_reused_and_invalidated(ko_lex, tmp_path, settings):
    from app.retrieval import HybridRetriever
    lex_file = settings.lexicon_dir / "ko.jsonl"
    stem = tmp_path / "ko_index"
    emb = CharGramEmbedder()
    HybridRetriever(ko_lex, emb, stem, lex_file)
    assert stem.with_suffix(".faiss").exists() and emb.calls == 1
    HybridRetriever(ko_lex, emb, stem, lex_file)  # same lexicon + model -> loaded, not re-encoded
    assert emb.calls == 1
    changed = tmp_path / "ko.jsonl"
    changed.write_bytes(lex_file.read_bytes() + b"\n")
    HybridRetriever(ko_lex, emb, stem, changed)  # lexicon file changed -> rebuilt
    assert emb.calls == 2


def test_pipeline_with_hybrid_retriever(settings, tmp_path):
    import dataclasses
    s = dataclasses.replace(settings, data_dir=tmp_path)
    import shutil
    shutil.copytree(settings.lexicon_dir, tmp_path / "lexicon")
    p = Pipeline(s, embedder=CharGramEmbedder())
    r = p.analyze("ko", ["아니 진짜 여기 분위기 대박이야."])
    assert r.meta.retriever == "hybrid" and [x.status for x in r.lines[0].spans] == ["explained", "explained"]
    assert (tmp_path / "assets" / "ko_index.faiss").exists()


# --------------------------------------------------------------------------- novelty signal
def _novelty_detector(lang, lex=()):
    from app.vocab import GeneralFreq, LearnerVocab
    root = Settings.from_env().wordlists_dir
    return Detector(lang, list(lex), None, 2.0, None,
                    LearnerVocab.load(root / f"{lang}_learner.tsv", 99), ("lexicon", "novelty"),
                    GeneralFreq.load(root / f"{lang}_general.tsv"), {"ko": 3.0, "zh": 2.5}[lang])


def test_novelty_flags_neologisms_not_standard_words():
    d = _novelty_detector("ko")
    flagged = {s.surface for s in d.detect("팀플 데이빗이 캐리 중인데 알잘딱깔센이라 넘나 고마운 맘ㅜㅜ 취뽀하면 고기나 때리러 가자")}
    assert {"팀플", "알잘딱깔센", "취뽀"} <= flagged
    assert "데이빗" not in flagged  # established (a name wordfreq knows)
    flagged = {s.surface for s in d.detect("수포자한테 알고리즘 수업은 넘나 난이도 높은 것..그래도 중꺽마!")}
    assert "중꺽마" in flagged and not {"알고리즘", "난이도"} & flagged
    assert all(s.detected_by == ["novelty"] for s in d.detect("중꺽마!"))


def test_novelty_ignores_everyday_lines_and_zh_standard_words():
    assert _novelty_detector("ko").detect("안녕하세요 여러분, 오늘은 성수동에 있는 카페에 왔어요.") == []
    zh = {s.surface for s in _novelty_detector("zh").detect("数学和算法都很难，今天我们去吃网红火锅")}
    assert not {"数学", "算法", "网红"} & zh


def test_novelty_yields_to_lexicon_on_overlap(ko_lex):
    spans = _novelty_detector("ko", ko_lex).detect("와 이거 존맛탱이다 ㄹㅇ.")
    hit = next(s for s in spans if s.surface == "존맛탱")
    assert hit.type == "slang" and "lexicon" in hit.detected_by
