from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.detector import ContrastTable, Detector, ngram_keys, segment
from app.lexicon import load_lexicon
from app.pipeline import Pipeline
from app.retrieval import dice


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    import dataclasses

    # no key: tests never hit the network
    return dataclasses.replace(Settings.from_env(), anthropic_api_key=None, explain_mode="auto")


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


def test_claude_rejection_marks_uncertain(settings, monkeypatch):
    """If Claude says the entries don't fit the context, the span becomes uncertain."""
    p = Pipeline(settings)
    ex = p.langs["ko"].explainer
    monkeypatch.setattr(ex, "api_key", "test")
    fake = SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={"items": [
        {"span_index": 0, "fits_context": False}]})])
    monkeypatch.setattr(ex, "_client", SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: fake)))
    r = p.analyze("ko", ["헐, 이 케이크 좀 보세요."])
    s = r.lines[0].spans[0]
    assert s.status == "uncertain" and s.explanation_source == "live"
    assert r.meta.mode == "live"


def test_api_failure_falls_back(settings, monkeypatch):
    p = Pipeline(settings)
    ex = p.langs["ko"].explainer
    monkeypatch.setattr(ex, "api_key", "test")

    def boom(**kw):
        raise RuntimeError("no network")

    monkeypatch.setattr(ex, "_client", SimpleNamespace(messages=SimpleNamespace(create=boom)))
    r = p.analyze("ko", ["헐, 이 케이크 좀 보세요."])
    assert r.lines[0].spans[0].status == "explained"
    assert r.meta.warnings and r.meta.mode in ("offline", "cache")


# --------------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(monkeypatch_module):
    from app.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    mp.delenv("ANTHROPIC_API_KEY", raising=False)
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
