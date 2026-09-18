"""Documents, chunks and the corpus — the thing mirage mutates."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit. `content_hash` is what the embedding cache is keyed on."""
    id: str
    doc_id: str
    text: str

    @property
    def content_hash(self) -> str:
        return _hash(self.text)


@dataclass
class Corpus:
    """A mutable set of chunks. Mutations are applied here and always undone."""
    chunks: dict[str, Chunk] = field(default_factory=dict)

    @classmethod
    def from_documents(cls, docs: dict[str, str], max_chars: int = 480) -> "Corpus":
        c = cls()
        for doc_id, text in docs.items():
            for i, piece in enumerate(chunk_text(text, max_chars)):
                cid = f"{doc_id}#{i}"
                c.chunks[cid] = Chunk(id=cid, doc_id=doc_id, text=piece)
        return c

    # -- mutation surface -------------------------------------------------
    def snapshot(self) -> dict[str, Chunk]:
        return dict(self.chunks)

    def restore(self, snap: dict[str, Chunk]) -> None:
        self.chunks = dict(snap)

    def replace_text(self, chunk_id: str, new_text: str) -> None:
        old = self.chunks[chunk_id]
        self.chunks[chunk_id] = replace(old, text=new_text)

    def delete(self, chunk_id: str) -> None:
        self.chunks.pop(chunk_id, None)

    def add(self, chunk: Chunk) -> None:
        self.chunks[chunk.id] = chunk

    def ids(self) -> list[str]:
        return sorted(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)


_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 480) -> list[str]:
    """Sentence-aware chunking. Kept simple and deterministic on purpose:
    chunking strategy is a variable mirage is used to *evaluate*, not to hide."""
    out: list[str] = []
    buf = ""
    for sentence in _SENT.split(text.strip()):
        if not sentence:
            continue
        if buf and len(buf) + len(sentence) + 1 > max_chars:
            out.append(buf.strip())
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf.strip():
        out.append(buf.strip())
    return out
