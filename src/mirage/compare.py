"""Did the answer change? A ladder, not a string comparison.

Two correct answers can be worded differently, so equality is wrong. The
threshold that decides "different" is NOT hardcoded — it is derived from the
pipeline's own measured variance (see calibrate.py).
"""
from __future__ import annotations

import re

import numpy as np

from .embed import HashingEmbedder

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_ABSTAIN = ("i don't know", "i do not know", "not in the context", "cannot answer",
            "no information", "unable to answer")


def is_abstention(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _ABSTAIN)


def numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in _NUM.findall(text)]


class AnswerComparer:
    """similarity() returns 1.0 for identical, lower as answers diverge.

    Ladder: exact -> abstention flip -> numeric/entity change -> embedding cosine.
    Each rung is cheaper and more certain than the next.
    """

    def __init__(self) -> None:
        self._e = HashingEmbedder()

    def similarity(self, a: str, b: str) -> float:
        a_s, b_s = a.strip().lower(), b.strip().lower()
        if a_s == b_s:
            return 1.0
        # An abstention on one side only is the strongest possible difference.
        if is_abstention(a) != is_abstention(b):
            return 0.0
        na, nb = numbers(a), numbers(b)
        if na and nb and na != nb:
            return 0.0
        va, vb = self._e.embed([a, b])
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
        return float(np.clip(float(va @ vb) / denom, 0.0, 1.0))
