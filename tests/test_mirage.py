"""mirage's own suite. Most of these are regressions for defects found by running it."""
import json
import random
from pathlib import Path

import pytest

from mirage.calibrate import Calibration, calibrate, refine
from mirage.compare import AnswerComparer, is_abstention, numbers
from mirage.corpus import Corpus, chunk_text
from mirage.embed import CachedEmbedder, HashingEmbedder
from mirage.engine import run
from mirage.evidence import resolve
from mirage.llm import ReplayModel
from mirage.operators import CONTROLS, WEIGHT, Kind, Mutant, apply, plan
from mirage.outcomes import Outcome
from mirage.pipeline import ReferencePipeline
from mirage.retrieve import BM25

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "false_confidence"


def _project(ungrounded=False):
    docs = json.loads((FIX / "corpus.json").read_text())
    qs = json.loads((FIX / "questions.json").read_text())
    par = json.loads((FIX / "parametric.json").read_text()) if ungrounded else {}
    return Corpus.from_documents(docs), qs, ReferencePipeline(model=ReplayModel(par))


# ---- corpus ---------------------------------------------------------------

def test_chunking_is_sentence_aware_and_lossless():
    text = "One sentence here. Another sentence follows. And a third one."
    assert " ".join(chunk_text(text, 40)).replace("  ", " ").count("sentence") == 2


def test_content_hash_changes_with_text():
    c = Corpus.from_documents({"d": "Alpha beta gamma."})
    cid = c.ids()[0]
    before = c.chunks[cid].content_hash
    c.replace_text(cid, "Alpha beta delta.")
    assert c.chunks[cid].content_hash != before


def test_snapshot_restore_undoes_every_mutation():
    c = Corpus.from_documents({"d": "Alpha. Beta. Gamma.", "e": "Delta. Epsilon."})
    snap = c.snapshot()
    assert len(c) >= 2
    c.delete(c.ids()[0])
    c.replace_text(c.ids()[0], "changed")
    c.restore(snap)
    assert c.snapshot() == snap


# ---- embedding cache ------------------------------------------------------

def test_embedding_cache_reuses_unchanged_chunks():
    e = CachedEmbedder(HashingEmbedder())
    items = [("h1", "alpha"), ("h2", "beta")]
    e.embed_chunks(items)
    assert (e.hits, e.misses) == (0, 2)
    e.embed_chunks(items + [("h3", "gamma")])
    assert (e.hits, e.misses) == (2, 3), "only the new chunk should be embedded"


# ---- retrieval ------------------------------------------------------------

def test_bm25_ranks_the_matching_document_first():
    b = BM25({"a": "the pump flow rate is high", "b": "annual leave policy"})
    assert max(b.scores("pump flow rate"), key=lambda k: b.scores("pump flow rate")[k]) == "a"


# ---- comparison ladder ----------------------------------------------------

def test_abstention_flip_is_maximum_difference():
    c = AnswerComparer()
    assert c.similarity("The revenue was 412 million.", "I don't know.") == 0.0


def test_changed_number_is_maximum_difference():
    c = AnswerComparer()
    assert c.similarity("Revenue was 412 million.", "Revenue was 831 million.") == 0.0


def test_identical_answers_are_identical():
    assert AnswerComparer().similarity("Same text.", "same text.") == 1.0


def test_numbers_strips_thousand_separators():
    assert numbers("Revenue was 1,412 million") == ["1412"]


def test_is_abstention_detects_the_phrases():
    assert is_abstention("I don't know.")
    assert not is_abstention("The answer is 42.")


# ---- calibration ----------------------------------------------------------

def test_calibration_of_a_deterministic_pipeline_has_zero_variance():
    corpus, qs, pipe = _project()
    pipe.index(corpus)
    cal = calibrate(pipe, qs, k=3)
    assert cal.deterministic and cal.stdev == 0.0


def test_refine_lowers_the_threshold_below_the_paraphrase_floor():
    """REGRESSION. The first version calibrated only on self-consistency, so the
    threshold sat at 0.98 and every paraphrase counted as a real change — the
    paraphrase control failed at 50% and the run was inconclusive."""
    base = Calibration(k=5, mean_self_similarity=1.0, stdev=0.0, threshold=0.98,
                       deterministic=True)
    out = refine(base, [0.94, 0.91, 0.97])
    assert out.threshold < 0.91
    assert out.paraphrase_floor == 0.91
    assert "paraphrase" in out.source


def test_refine_without_paraphrase_data_is_a_no_op():
    base = Calibration(5, 1.0, 0.0, 0.98, True)
    assert refine(base, []) is base


# ---- evidence sets --------------------------------------------------------

def test_evidence_set_finds_every_chunk_stating_the_fact():
    """The multi-source confound: mutating only the top chunk would leave the
    duplicate intact and record a false survivor."""
    corpus = Corpus.from_documents({
        "a": "Northwind revenue was 412 million dollars in 2025.",
        "b": "As reported elsewhere, Northwind revenue was 412 million dollars in 2025.",
        "c": "Unrelated text about shipping containers.",
    })
    ev = resolve(corpus, "What was Northwind revenue?",
                 "Northwind revenue was 412 million dollars in 2025.", corpus.ids()[:1])
    assert len(ev.chunk_ids) == 2, "both chunks carrying the figure are evidence"


def test_evidence_falls_back_to_top_retrieved_and_says_so():
    corpus = Corpus.from_documents({"a": "Completely unrelated content."})
    ev = resolve(corpus, "q", "zzz qqq", ["a#0"])
    assert ev.chunk_ids == ["a#0"] and "fallback" in ev.method


def test_no_answer_yields_no_evidence():
    corpus = Corpus.from_documents({"a": "text"})
    assert not resolve(corpus, "q", "   ", ["a#0"])


# ---- operators ------------------------------------------------------------

def test_every_operator_family_is_planned_when_evidence_exists():
    corpus, qs, pipe = _project()
    pipe.index(corpus)
    a = pipe.answer(qs[0])
    ev = resolve(corpus, qs[0], a.text, a.retrieved)
    kinds = {m.kind for m in plan(corpus, qs[0], ev, random.Random(1), [0])}
    assert kinds == set(Kind), "all seven operators, including both controls"


def test_controls_carry_zero_weight_and_are_flagged():
    assert CONTROLS == {Kind.PARAPHRASE, Kind.IRRELEVANT}
    for k in CONTROLS:
        assert WEIGHT[k] == 0.0
    m = Mutant("M1", Kind.PARAPHRASE, "q", ["a#0"], "d")
    assert m.is_control


def test_deletion_is_weighted_highest():
    assert WEIGHT[Kind.EVIDENCE_DELETION] == max(WEIGHT.values())


def test_fact_corruption_changes_every_number():
    corpus = Corpus.from_documents({"a": "Revenue was 412 million over 38 sites."})
    cid = corpus.ids()[0]
    apply(corpus, Mutant("M1", Kind.FACT_CORRUPTION, "q", [cid], "d"), random.Random(1))
    assert "412" not in corpus.chunks[cid].text and "38" not in corpus.chunks[cid].text


def test_evidence_deletion_removes_the_whole_set():
    corpus = Corpus.from_documents({"a": "One. Two.", "b": "Three."})
    ids = corpus.ids()
    apply(corpus, Mutant("M1", Kind.EVIDENCE_DELETION, "q", ids[:2], "d"), random.Random(1))
    assert len(corpus) == len(ids) - 2


def test_distractor_and_contradiction_add_rather_than_replace():
    corpus = Corpus.from_documents({"a": "Revenue was 412 million."})
    cid = corpus.ids()[0]
    before = len(corpus)
    apply(corpus, Mutant("M1", Kind.DISTRACTOR, "q", [cid], "d"), random.Random(1))
    apply(corpus, Mutant("M2", Kind.CONTRADICTION, "q", [cid], "d"), random.Random(1))
    assert len(corpus) == before + 2
    assert corpus.chunks[cid].text == "Revenue was 412 million."


def test_non_viable_mutation_reports_false():
    corpus = Corpus.from_documents({"a": "no numbers or names here"})
    cid = corpus.ids()[0]
    assert apply(corpus, Mutant("M1", Kind.FACT_CORRUPTION, "q", [cid], "d"),
                 random.Random(1)) is False


# ---- end to end -----------------------------------------------------------

def test_grounded_pipeline_passes_controls_and_scores_high():
    corpus, qs, pipe = _project()
    r = run(pipe, corpus, qs, k=3)
    assert r.controls_pass(), "a faithful reader must hold its answer under both controls"
    assert not r.inconclusive()
    assert r.sensitivity() >= 0.6
    assert r.by_operator()[Kind.EVIDENCE_DELETION][0] == r.by_operator()[Kind.EVIDENCE_DELETION][1]


def test_ungrounded_pipeline_survives_evidence_deletion():
    """The false-confidence demo, as a test. A model answering from parametric
    memory keeps answering after its evidence is deleted."""
    corpus, qs, pipe = _project(ungrounded=True)
    r = run(pipe, corpus, qs, k=3)
    killed, total = r.by_operator()[Kind.EVIDENCE_DELETION]
    assert killed < total, "deletion survivors are the finding"
    assert any(s.mutant.kind is Kind.EVIDENCE_DELETION for s in r.survivors())


def test_ungrounded_scores_strictly_lower_than_grounded():
    c1, qs, p1 = _project()
    c2, _, p2 = _project(ungrounded=True)
    assert run(p2, c2, qs, k=3).sensitivity() < run(p1, c1, qs, k=3).sensitivity()


def test_the_corpus_is_restored_after_a_run():
    corpus, qs, pipe = _project()
    before = corpus.snapshot()
    run(pipe, corpus, qs, k=3)
    assert corpus.snapshot() == before, "mutations must never leak out of a run"


def test_errors_are_reported_not_hidden():
    class Broken(ReferencePipeline):
        def answer(self, question):
            raise RuntimeError("boom")
    corpus, qs, _ = _project()
    r = run(Broken(), corpus, [qs[0]], k=1)
    assert all(x.outcome is Outcome.ERROR for x in r.results if x.outcome is not Outcome.NOT_VIABLE)
    assert r.inconclusive(), "a run of nothing but errors cannot produce a score"
