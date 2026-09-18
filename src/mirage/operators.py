"""The seven mutation operators, including two controls."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum

from .corpus import Chunk, Corpus
from .evidence import EvidenceSet


class Kind(str, Enum):
    FACT_CORRUPTION = "fact_corruption"
    ENTITY_SWAP = "entity_swap"
    EVIDENCE_DELETION = "evidence_deletion"
    CONTRADICTION = "contradiction"
    DISTRACTOR = "distractor"
    PARAPHRASE = "paraphrase"              # positive control: answer should HOLD
    IRRELEVANT = "irrelevant_corruption"   # negative control: answer should HOLD


CONTROLS = {Kind.PARAPHRASE, Kind.IRRELEVANT}

# How much a survivor of each operator tells you. Deletion is strictest: with the
# evidence gone there is no other document the answer could have come from.
WEIGHT = {
    Kind.EVIDENCE_DELETION: 1.0,
    Kind.FACT_CORRUPTION: 0.9,
    Kind.ENTITY_SWAP: 0.85,
    Kind.CONTRADICTION: 0.6,
    Kind.DISTRACTOR: 0.4,
    Kind.PARAPHRASE: 0.0,
    Kind.IRRELEVANT: 0.0,
}


@dataclass
class Mutant:
    id: str
    kind: Kind
    question: str
    targets: list[str]
    description: str

    @property
    def is_control(self) -> bool:
        return self.kind in CONTROLS


_NUM = re.compile(r"(?<![\w.])(\d[\d,]*\.?\d*)(?![\w.])")
_CAP = re.compile(r"\b([A-Z][a-z]{2,})\b")


def _bump_numbers(text: str) -> str | None:
    def rep(m: re.Match[str]) -> str:
        raw = m.group(1).replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            return m.group(0)
        new = v * 2 + 7
        return str(int(new)) if v.is_integer() else f"{new:.2f}"
    out, n = _NUM.subn(rep, text)
    return out if n else None


def _swap_entities(text: str, rng: random.Random) -> str | None:
    names = sorted(set(_CAP.findall(text)))
    if not names:
        return None
    fake = ["Zephyrix", "Calloway", "Brantmoor", "Vexley", "Ordanis"]
    mapping = {n: fake[i % len(fake)] for i, n in enumerate(names)}
    out = text
    for a, b in mapping.items():
        out = re.sub(rf"\b{re.escape(a)}\b", b, out)
    return out if out != text else None


def _paraphrase(text: str) -> str:
    subs = [(r"\bis\b", "remains"), (r"\bwas\b", "had been"), (r"\bhas\b", "possesses"),
            (r"\bare\b", "remain"), (r"\buses\b", "employs"), (r"\bmust\b", "is required to")]
    out = text
    for a, b in subs:
        out = re.sub(a, b, out, count=1)
    return f"To restate: {out}" if out == text else out


def apply(corpus: Corpus, mutant: Mutant, rng: random.Random) -> bool:
    """Mutate in place. Returns False when the mutation is not viable here."""
    k = mutant.kind
    if k is Kind.EVIDENCE_DELETION:
        for cid in mutant.targets:
            corpus.delete(cid)
        return True
    if k is Kind.DISTRACTOR:
        src = corpus.chunks.get(mutant.targets[0])
        if src is None:
            return False
        near = _bump_numbers(src.text) or _swap_entities(src.text, rng)
        if near is None:
            return False
        corpus.add(Chunk(id=f"{src.id}~distractor", doc_id=src.doc_id, text=near))
        return True
    if k is Kind.CONTRADICTION:
        src = corpus.chunks.get(mutant.targets[0])
        if src is None:
            return False
        corpus.add(Chunk(id=f"{src.id}~contra", doc_id=src.doc_id,
                         text=f"Correction. The following is not accurate: {src.text} "
                              f"The opposite is the case."))
        return True

    changed = False
    for cid in mutant.targets:
        chunk = corpus.chunks.get(cid)
        if chunk is None:
            continue
        if k is Kind.FACT_CORRUPTION:
            new = _bump_numbers(chunk.text)
        elif k is Kind.ENTITY_SWAP:
            new = _swap_entities(chunk.text, rng)
        elif k in (Kind.PARAPHRASE,):
            new = _paraphrase(chunk.text)
        elif k is Kind.IRRELEVANT:
            new = _bump_numbers(chunk.text) or _swap_entities(chunk.text, rng) or _paraphrase(chunk.text)
        else:
            new = None
        if new and new != chunk.text:
            corpus.replace_text(cid, new)
            changed = True
    return changed


def plan(corpus: Corpus, question: str, evidence: EvidenceSet,
         rng: random.Random, counter: list[int]) -> list[Mutant]:
    """One mutant per applicable operator for this question."""
    out: list[Mutant] = []

    def nxt(kind: Kind, targets: list[str], desc: str) -> None:
        counter[0] += 1
        out.append(Mutant(f"M{counter[0]:03d}", kind, question, targets, desc))

    ev = evidence.chunk_ids
    if ev:
        nxt(Kind.EVIDENCE_DELETION, list(ev), f"remove all {len(ev)} evidence chunk(s) from the index")
        nxt(Kind.FACT_CORRUPTION, list(ev), "alter every number in the evidence")
        nxt(Kind.ENTITY_SWAP, list(ev), "replace named entities in the evidence")
        nxt(Kind.CONTRADICTION, [ev[0]], "inject a chunk contradicting the evidence")
        nxt(Kind.DISTRACTOR, [ev[0]], "inject a plausible near-miss chunk")
        nxt(Kind.PARAPHRASE, list(ev), "restate the evidence — answer should HOLD")

    # Negative control: something the question has nothing to do with.
    off = [c for c in corpus.ids() if c not in set(ev)]
    if off:
        nxt(Kind.IRRELEVANT, [rng.choice(off)], "corrupt an unrelated chunk — answer should HOLD")
    return out
