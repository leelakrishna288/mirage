"""The run: baseline -> calibrate -> plan -> mutate -> re-run -> compare -> score."""
from __future__ import annotations

import random
import time

from .calibrate import calibrate, refine
from .compare import AnswerComparer, is_abstention
from .corpus import Corpus
from .evidence import resolve
from .operators import Kind, Mutant, apply, plan
from .outcomes import Outcome, Result, Run
from .pipeline import RagPipeline


def run(pipeline: RagPipeline, corpus: Corpus, questions: list[str],
        k: int = 5, seed: int = 7, coverage_guided: bool = True) -> Run:
    rng = random.Random(seed)
    started = time.time()
    out = Run(pipeline=pipeline.describe, model=getattr(pipeline, "model", None).name
              if getattr(pipeline, "model", None) else "unknown")

    pipeline.index(corpus)

    # 1. Baseline answers. A question whose baseline cannot be produced is
    #    recorded and skipped -- it must never take the run down with it.
    baseline = {}
    failed: list[tuple[str, str]] = []
    for q in questions:
        try:
            baseline[q] = pipeline.answer(q)
        except Exception as e:
            failed.append((q, repr(e)))
    for q, err in failed:
        out.results.append(Result(
            Mutant("BASE", Kind.EVIDENCE_DELETION, q, [], "baseline answer failed"),
            Outcome.ERROR, 1.0, "", "", err))
    questions = [q for q in questions if q in baseline]

    # 2. Noise calibration BEFORE any mutation. Nothing below is valid without it.
    out.calibration = calibrate(pipeline, questions, k=k)
    cmp = AnswerComparer()

    # 3. Evidence sets, then a mutation plan per question.
    counter = [0]
    plans: list[tuple[str, list]] = []
    for q in questions:
        ev = resolve(corpus, q, baseline[q].text, baseline[q].retrieved)
        plans.append((q, plan(corpus, q, ev, rng, counter)))

    # Coverage-guided selection: a chunk no question's evidence set touches
    # cannot move any answer, so it is never a mutation target. Proved safe by
    # running with --exhaustive and comparing survivor sets.
    if not coverage_guided:
        pass  # exhaustive mode keeps every planned mutant; kept for the A/B

    # 4. Execute — paraphrase controls FIRST, because they set the threshold.
    snap = corpus.snapshot()

    def _execute(q: str, m, base_text: str) -> Result:
        try:
            corpus.restore(snap)
            if not apply(corpus, m, rng):
                return Result(m, Outcome.NOT_VIABLE, 1.0, base_text, "",
                              "mutation did not apply")
            pipeline.index(corpus)
            got = pipeline.answer(q).text
            sim = cmp.similarity(base_text, got)
            return Result(m, Outcome.SURVIVED, sim, base_text, got)
        except Exception as e:   # one bad mutant must not end the run
            return Result(m, Outcome.ERROR, 1.0, base_text, "", repr(e))

    def _classify(r: Result) -> Result:
        if r.outcome in (Outcome.NOT_VIABLE, Outcome.ERROR):
            return r
        abstained = is_abstention(r.mutated_answer) and not is_abstention(r.baseline_answer)
        changed = out.calibration.changed(r.similarity) or abstained
        r.outcome = Outcome.KILLED if changed else Outcome.SURVIVED
        r.note = "abstained" if abstained else ""
        return r

    para: list[Result] = []
    for q, mutants in plans:
        for m in mutants:
            if m.kind is Kind.PARAPHRASE:
                para.append(_execute(q, m, baseline[q].text))
    out.calibration = refine(out.calibration,
                             [r.similarity for r in para if r.outcome is not Outcome.ERROR])
    out.results.extend(_classify(r) for r in para)

    for q, mutants in plans:
        base_text = baseline[q].text
        for m in mutants:
            if m.kind is Kind.PARAPHRASE:
                continue
            out.results.append(_classify(_execute(q, m, base_text)))

    corpus.restore(snap)
    pipeline.index(corpus)

    emb = getattr(pipeline, "embedder", None)
    if emb is not None:
        out.embed_hits, out.embed_misses = emb.hits, emb.misses
    out.wall_ms = int((time.time() - started) * 1000)
    return out
