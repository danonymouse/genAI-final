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
    assert len(load_lexicon(settings.lexicon_dir / "ko.jsonl")) == 20
    assert len(load_lexicon(settings.lexicon_dir / "zh.jsonl")) >= 10


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


# --------------------------------------------------------------------------- stage 3
def test_dice():
    assert dice("대박이야", "대박이야") == 1.0
    assert dice("대박", "꿀잼") == 0.0
    assert 0 < dice("장난아니에요", "장난 아니에요") <= 1.0


# --------------------------------------------------------------------------- pipeline
def test_uncertain_when_nothing_retrieved(settings, tmp_path):
    p = Pipeline(settings)
    p.langs["zh"].detector.table = ContrastTable({"蜘蛛侠": [500, 1]}, 10**6, 10**6)
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
    assert r["status"] == "ok" and r["languages"]["ko"]["lexicon_entries"] == 20


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
