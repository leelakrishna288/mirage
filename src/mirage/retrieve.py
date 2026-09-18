"""Hybrid retrieval: BM25 (lexical) + dense (semantic), fused with RRF."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .corpus import Corpus
from .embed import CachedEmbedder

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = sorted(docs)
        self.toks = {i: tokenize(docs[i]) for i in self.ids}
        self.len = {i: len(self.toks[i]) for i in self.ids}
        self.avg = (sum(self.len.values()) / len(self.len)) if self.len else 0.0
        df: Counter[str] = Counter()
        for i in self.ids:
            df.update(set(self.toks[i]))
        n = max(1, len(self.ids))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = {i: Counter(self.toks[i]) for i in self.ids}

    def scores(self, query: str) -> dict[str, float]:
        q = tokenize(query)
        out: dict[str, float] = {}
        for i in self.ids:
            s, tf, dl = 0.0, self.tf[i], self.len[i]
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out[i] = s
        return out


@dataclass
class Retrieved:
    chunk_id: str
    text: str
    score: float


class HybridRetriever:
    """BM25 + dense, combined by Reciprocal Rank Fusion.

    RRF rather than score-weighted blending: the two scores are on different
    scales, and a weighted sum silently makes the weighting a hyperparameter
    nobody tuned.
    """

    def __init__(self, embedder: CachedEmbedder, k: int = 5, rrf_k: int = 60):
        self.embedder = embedder
        self.k = k
        self.rrf_k = rrf_k
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._bm25: BM25 | None = None

    def index(self, corpus: Corpus) -> None:
        self._ids = corpus.ids()
        items = [(corpus.chunks[i].content_hash, corpus.chunks[i].text) for i in self._ids]
        self._mat = self.embedder.embed_chunks(items)
        self._bm25 = BM25({i: corpus.chunks[i].text for i in self._ids})
        self._texts = {i: corpus.chunks[i].text for i in self._ids}

    def search(self, query: str, k: int | None = None) -> list[Retrieved]:
        k = k or self.k
        if not self._ids or self._mat is None or self._bm25 is None:
            return []
        qv = self.embedder.embed([query])[0]
        dense = self._mat @ qv
        dense_rank = {cid: r for r, cid in enumerate(
            sorted(self._ids, key=lambda c: -float(dense[self._ids.index(c)])))}
        lex = self._bm25.scores(query)
        lex_rank = {cid: r for r, cid in enumerate(sorted(self._ids, key=lambda c: -lex[c]))}
        fused = {
            cid: 1.0 / (self.rrf_k + dense_rank[cid]) + 1.0 / (self.rrf_k + lex_rank[cid])
            for cid in self._ids
        }
        top = sorted(self._ids, key=lambda c: -fused[c])[:k]
        return [Retrieved(c, self._texts[c], fused[c]) for c in top]
