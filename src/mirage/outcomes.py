"""Outcomes and scores — same taxonomy as mutagent, on purpose."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from .operators import CONTROLS, WEIGHT, Kind, Mutant


class Outcome(str, Enum):
    KILLED = "KILLED"          # answer moved, or the system abstained — grounded
    SURVIVED = "SURVIVED"      # unchanged despite corrupted evidence — the finding
    NOT_VIABLE = "NOT_VIABLE"  # mutation could not be applied
    ERROR = "ERROR"


@dataclass
class Result:
    mutant: Mutant
    outcome: Outcome
    similarity: float
    baseline_answer: str = ""
    mutated_answer: str = ""
    note: str = ""


@dataclass
class Run:
    results: list[Result] = field(default_factory=list)
    calibration: object | None = None
    pipeline: str = ""
    model: str = ""
    wall_ms: int = 0
    embed_hits: int = 0
    embed_misses: int = 0

    # -- scoring ----------------------------------------------------------
    def scored(self) -> list[Result]:
        return [r for r in self.results
                if not r.mutant.is_control and r.outcome in (Outcome.KILLED, Outcome.SURVIVED)]

    def sensitivity(self) -> float:
        s = self.scored()
        if not s:
            return 0.0
        return sum(1 for r in s if r.outcome is Outcome.KILLED) / len(s)

    def weighted_sensitivity(self) -> float:
        s = self.scored()
        tot = sum(WEIGHT[r.mutant.kind] for r in s)
        if not tot:
            return 0.0
        return sum(WEIGHT[r.mutant.kind] for r in s if r.outcome is Outcome.KILLED) / tot

    def by_operator(self) -> dict[Kind, tuple[int, int]]:
        agg: dict[Kind, list[int]] = defaultdict(lambda: [0, 0])
        for r in self.results:
            if r.outcome in (Outcome.KILLED, Outcome.SURVIVED):
                agg[r.mutant.kind][1] += 1
                if r.outcome is Outcome.KILLED:
                    agg[r.mutant.kind][0] += 1
        return {k: (v[0], v[1]) for k, v in agg.items()}

    def controls(self) -> dict[Kind, tuple[int, int]]:
        """For controls the answer SHOULD hold, so 'held' = SURVIVED."""
        agg: dict[Kind, list[int]] = defaultdict(lambda: [0, 0])
        for r in self.results:
            if r.mutant.kind in CONTROLS and r.outcome in (Outcome.KILLED, Outcome.SURVIVED):
                agg[r.mutant.kind][1] += 1
                if r.outcome is Outcome.SURVIVED:
                    agg[r.mutant.kind][0] += 1
        return {k: (v[0], v[1]) for k, v in agg.items()}

    def controls_pass(self, minimum: float = 0.8) -> bool:
        c = self.controls()
        if not c:
            return False
        return all(held / total >= minimum for held, total in c.values() if total)

    def inconclusive(self) -> bool:
        """No scored mutants, or the controls failed — report nothing rather than
        a number that cannot be trusted. Same discipline as mutagent."""
        return not self.scored() or not self.controls_pass()

    def survivors(self) -> list[Result]:
        return sorted((r for r in self.scored() if r.outcome is Outcome.SURVIVED),
                      key=lambda r: -WEIGHT[r.mutant.kind])
