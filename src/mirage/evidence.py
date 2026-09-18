"""Resolve the EVIDENCE SET behind an answer — every chunk carrying the claim.

Mutating only the top-retrieved chunk produces false survivors: another document
may state the same fact, so the answer legitimately survives. Resolving the set
is the engineering fix for that, not a softer claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from .compare import numbers
from .corpus import Corpus
from .retrieve import tokenize


@dataclass
class EvidenceSet:
    chunk_ids: list[str]
    primary: str | None
    method: str          # how it was resolved — printed in the report

    def __bool__(self) -> bool:
        return bool(self.chunk_ids)


def resolve(corpus: Corpus, question: str, answer: str, retrieved: list[str],
            overlap: float = 0.6) -> EvidenceSet:
    """A chunk is evidence if it was retrieved AND supports the answer, or if it
    states the same distinguishing facts (numbers, rare terms) as the answer."""
    if not answer.strip():
        return EvidenceSet([], None, "no answer")

    ans_nums = set(numbers(answer))
    ans_terms = {t for t in tokenize(answer) if len(t) > 3}
    q_terms = set(tokenize(question))
    # Terms that distinguish the answer from the question — the answer's payload.
    payload = ans_terms - q_terms

    hits: list[str] = []
    for cid, chunk in corpus.chunks.items():
        c_nums = set(numbers(chunk.text))
        c_terms = {t for t in tokenize(chunk.text) if len(t) > 3}
        if ans_nums and ans_nums & c_nums:
            hits.append(cid)
            continue
        if payload and len(payload & c_terms) / len(payload) >= overlap:
            hits.append(cid)

    if not hits:
        # Fall back to the top retrieved chunk, and say so in the report.
        top = retrieved[0] if retrieved else None
        return EvidenceSet([top] if top else [], top, "top-retrieved (fallback)")

    primary = next((c for c in retrieved if c in hits), hits[0])
    method = "numeric + term overlap" if ans_nums else "term overlap"
    return EvidenceSet(sorted(hits), primary, method)
