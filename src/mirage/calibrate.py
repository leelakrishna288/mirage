"""Measure the pipeline's own answer variance BEFORE mutating anything.

Nothing downstream is interpretable without this. If the same question, asked
of the unmutated corpus k times, already produces answers that differ by 0.3,
then a mutant that moves the answer by 0.2 has told you nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from .compare import AnswerComparer
from .pipeline import RagPipeline


@dataclass
class Calibration:
    k: int
    mean_self_similarity: float
    stdev: float
    threshold: float          # answers must fall BELOW this to count as changed
    deterministic: bool
    paraphrase_floor: float | None = None   # set by refine(); see below
    source: str = "self-consistency"

    def changed(self, similarity: float) -> bool:
        return similarity < self.threshold


def refine(base: Calibration, paraphrase_sims: list[float], margin: float = 0.02) -> Calibration:
    """Second calibration stage — the one that matters.

    Self-consistency alone only measures sampling noise. It says nothing about
    legitimate *rewording*: an extractive reader will echo whatever wording the
    source uses, so paraphrasing the evidence changes the answer's words while
    the meaning holds. Scored against a self-consistency threshold, every
    paraphrase looks like a real change and the control fails.

    So the paraphrase control sets the bar: an answer counts as changed only if
    it moves FURTHER than a paraphrase of its own evidence moves it. Found by
    running the tool — the first version failed its own control at 50%.
    """
    if not paraphrase_sims:
        return base
    floor = min(paraphrase_sims)
    return Calibration(
        k=base.k,
        mean_self_similarity=base.mean_self_similarity,
        stdev=base.stdev,
        threshold=max(0.0, min(base.threshold, floor - margin)),
        deterministic=base.deterministic,
        paraphrase_floor=floor,
        source="self-consistency + paraphrase control",
    )


def calibrate(pipeline: RagPipeline, questions: list[str], k: int = 5,
              margin: float = 3.0, floor: float = 0.02) -> Calibration:
    cmp = AnswerComparer()
    sims: list[float] = []
    for q in questions:
        try:
            runs = [pipeline.answer(q).text for _ in range(k)]
        except Exception:
            continue          # a question that cannot be answered cannot be calibrated
        base = runs[0]
        sims.extend(cmp.similarity(base, r) for r in runs[1:])
    if not sims:
        return Calibration(k, 1.0, 0.0, 1.0 - floor, True)
    m, sd = mean(sims), (pstdev(sims) if len(sims) > 1 else 0.0)
    # A change must exceed the pipeline's own wobble by `margin` standard
    # deviations, with a floor so a perfectly deterministic pipeline still
    # requires a real (not floating-point) difference.
    threshold = min(1.0, max(0.0, m - max(margin * sd, floor)))
    return Calibration(k, m, sd, threshold, sd == 0.0)
