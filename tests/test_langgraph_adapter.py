"""The LangGraph adapter must measure the SAME pipeline the reference does.

These tests skip cleanly when the optional extra is absent, so the core suite
still runs with mirage's two hard dependencies and no network.
"""
import json
from pathlib import Path

import pytest

from mirage.corpus import Corpus
from mirage.engine import run
from mirage.llm import ReplayModel
from mirage.pipeline import ReferencePipeline

langgraph = pytest.importorskip("langgraph", reason="optional extra: .[langgraph]")
from mirage.adapters.langgraph_pipeline import LangGraphPipeline  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "false_confidence"


def _project(ungrounded=False):
    docs = json.loads((FIX / "corpus.json").read_text())
    qs = json.loads((FIX / "questions.json").read_text())
    par = json.loads((FIX / "parametric.json").read_text()) if ungrounded else {}
    return Corpus.from_documents(docs), qs, ReplayModel(par)


def test_adapter_satisfies_the_pipeline_protocol():
    _, _, model = _project()
    p = LangGraphPipeline(model=model)
    assert hasattr(p, "index") and hasattr(p, "answer") and hasattr(p, "describe")
    assert "langgraph" in p.describe


def test_graph_answers_identically_to_the_reference_pipeline():
    """The claim that matters. If these ever diverge, a measurement taken
    through one pipeline would not describe the other."""
    corpus, qs, model = _project()
    ref = ReferencePipeline(model=ReplayModel()); ref.index(corpus)
    lg = LangGraphPipeline(model=model); lg.index(corpus)
    for q in qs:
        a, b = ref.answer(q), lg.answer(q)
        assert a.text == b.text, f"answers diverge for {q!r}"
        assert a.retrieved == b.retrieved, f"retrieval diverges for {q!r}"


def test_a_full_run_through_the_graph_scores_the_same():
    """End to end: the sensitivity score must not depend on which pipeline
    implementation produced the answers."""
    c1, qs, m1 = _project()
    c2, _, m2 = _project()
    ref_run = run(ReferencePipeline(model=m1), c1, qs, k=3)
    lg_run = run(LangGraphPipeline(model=m2), c2, qs, k=3)
    assert ref_run.sensitivity() == lg_run.sensitivity()
    assert ref_run.controls_pass() and lg_run.controls_pass()
    assert ref_run.by_operator() == lg_run.by_operator()


def test_ungrounded_is_caught_through_the_graph_too():
    corpus, qs, model = _project(ungrounded=True)
    r = run(LangGraphPipeline(model=model), corpus, qs, k=3)
    from mirage.operators import Kind
    killed, total = r.by_operator()[Kind.EVIDENCE_DELETION]
    assert killed < total, "deletion survivors must be found through the graph too"


def test_missing_extra_raises_rather_than_falling_back(monkeypatch):
    """A run must never silently measure a different pipeline from the one it
    names in its report."""
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name.startswith("langgraph"):
            raise ImportError("blocked for the test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="optional extra"):
        LangGraphPipeline()
