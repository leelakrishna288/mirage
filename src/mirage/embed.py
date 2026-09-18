"""Embeddings, behind a protocol, with a content-hash cache.

The cache is the direct analogue of mutagent's compilation cache: mutating one
chunk must not force the whole corpus to be re-embedded.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    @property
    def name(self) -> str: ...


class HashingEmbedder:
    """Deterministic, offline, no model download. Good enough to demonstrate the
    method and — being deterministic — it keeps the pipeline's own noise at zero
    on the retrieval side, so §calibration isolates *model* variance."""

    def __init__(self, dim: int = 512) -> None:
        self._v = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm="l2", stop_words="english"
        )
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self._v.transform(texts).toarray().astype(np.float32)


class CachedEmbedder:
    """Wraps any Embedder. Keyed on (embedder name, chunk content hash)."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def embed_chunks(self, items: list[tuple[str, str]]) -> np.ndarray:
        """items: (content_hash, text). Only unseen hashes reach the inner embedder."""
        need = [(h, t) for h, t in items if h not in self._cache]
        self.hits += len(items) - len(need)
        self.misses += len(need)
        if need:
            vecs = self._inner.embed([t for _, t in need])
            for (h, _), v in zip(need, vecs):
                self._cache[h] = v
        return np.vstack([self._cache[h] for h, _ in items]) if items else self._inner.embed([])

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._inner.embed(texts)
