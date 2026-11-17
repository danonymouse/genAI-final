"""Wire stages 1-4 together. One `Pipeline` per process, holding per-language components."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .detector import ContrastTable, Detector
from .explain import Explainer, SpanJob
from .llm import OpenAICompatClient
from .lexicon import load_lexicon
from .retrieval import KeywordRetriever, Retriever
from .schemas import AnalyzedLine, AnalyzedSpan, AnalyzeResponse, Lang, LexiconEntry, Meta

LANGS: tuple[Lang, ...] = ("ko", "zh")
CONTEXT_WINDOW = 2  # lines before and after sent as context


@dataclass
class LangComponents:
    lexicon: list[LexiconEntry]
    detector: Detector
    retriever: Retriever
    explainer: Explainer


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = s = settings or Settings.from_env()
        self.langs: dict[Lang, LangComponents] = {}
        llm = (OpenAICompatClient(s.llm_base_url, s.llm_model, s.llm_api_key, s.llm_timeout)
               if s.live_enabled and s.llm_base_url else None)
        for lang in LANGS:
            lex = load_lexicon(s.lexicon_dir / f"{lang}.jsonl")
            table = ContrastTable.load(s.assets_dir / f"{lang}_freq.json.gz")
            self.langs[lang] = LangComponents(
                lexicon=lex,
                detector=Detector(lang, lex, table, s.contrast_threshold),
                retriever=KeywordRetriever(lex),
                explainer=Explainer(
                    lang=lang,
                    cache_path=s.cache_dir / f"{lang}.json",
                    llm=llm,
                    use_cache=s.explain_mode in ("auto", "cache"),
                ),
            )

    def analyze(self, lang: Lang, lines: list[str]) -> AnalyzeResponse:
        c = self.langs[lang]
        thr = self.settings.retrieval_threshold
        c.explainer.warnings = []
        out_lines: list[AnalyzedLine] = []
        sources: set[str] = set()

        for idx, line in enumerate(lines):
            flagged = c.detector.detect(line)  # stages 1-2
            analyzed: list[AnalyzedSpan | None] = [None] * len(flagged)
            jobs: list[SpanJob] = []
            job_pos: list[int] = []

            for i, span in enumerate(flagged):  # stage 3
                hits = c.retriever.search(span, line, k=3)
                matches = [m for m, _ in hits]
                good = [(m, e) for m, e in hits if m.similarity >= thr]
                if not good:
                    best = matches[0].similarity if matches else 0.0
                    analyzed[i] = AnalyzedSpan(
                        **span.model_dump(), status="uncertain", matches=matches,
                        uncertain_reason=f"No lexicon entry matched closely enough "
                                         f"(best similarity {best:.2f} < {thr:.2f}), "
                                         f"so no explanation was generated.",
                    )
                    continue
                if span.type is None:  # contrast-only flag: take the type from the best match
                    span = span.model_copy(update={"type": good[0][0].type})
                    flagged[i] = span
                analyzed[i] = AnalyzedSpan(**span.model_dump(), status="explained", matches=matches)
                jobs.append(SpanJob(span=span, entries=[e for _, e in good]))
                job_pos.append(i)

            context = lines[max(0, idx - CONTEXT_WINDOW): idx] + lines[idx + 1: idx + 1 + CONTEXT_WINDOW]
            results = c.explainer.explain_line(line, context, jobs)  # stage 4
            for pos, res in zip(job_pos, results):
                a = analyzed[pos]
                assert a is not None
                if res.rejected:
                    a.status = "uncertain"
                    a.uncertain_reason = "The retrieved entries do not fit how the phrase is used here."
                else:
                    a.explanation = res.explanation
                a.explanation_source = res.source  # type: ignore[assignment]
                if res.source:
                    sources.add(res.source)

            out_lines.append(AnalyzedLine(line_id=idx, text=line, spans=[a for a in analyzed if a]))

        all_spans = [s for ln in out_lines for s in ln.spans]
        mode = "live" if "live" in sources else "cache" if "cache" in sources else "offline"
        return AnalyzeResponse(
            lang=lang,
            lines=out_lines,
            meta=Meta(
                mode=mode,
                model=self.settings.llm_model if mode == "live" else None,
                detector=c.detector.name,
                retriever=c.retriever.name,
                retrieval_threshold=thr,
                n_flagged=len(all_spans),
                n_uncertain=sum(s.status == "uncertain" for s in all_spans),
                warnings=list(c.explainer.warnings),
            ),
        )
