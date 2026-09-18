"""The RagPipeline protocol and the in-repo reference implementation.

The protocol is the point: mirage must not be "a LangGraph evaluator". Anything
that can answer a question and say which chunks it retrieved can be measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .corpus import Corpus
from .embed import CachedEmbedder, HashingEmbedder
from .llm import ChatModel, ReplayModel
from .retrieve import HybridRetriever

PROMPT = """Answer the question using ONLY the context below. If the context does
not contain the answer, say "I don't know."

CONTEXT:
{context}

QUESTION:
{question}
"""


@dataclass
class Answer:
    text: str
    retrieved: list[str] = field(default_factory=list)   # chunk ids, best first


class RagPipeline(Protocol):
    def index(self, corpus: Corpus) -> None: ...
    def answer(self, question: str) -> Answer: ...
    @property
    def describe(self) -> str: ...


class ReferencePipeline:
    """Chunk -> embed -> hybrid retrieve -> prompt -> answer. Deliberately plain."""

    def __init__(self, model: ChatModel | None = None, k: int = 4):
        self.embedder = CachedEmbedder(HashingEmbedder())
        self.retriever = HybridRetriever(self.embedder, k=k)
        self.model = model or ReplayModel()
        self.k = k

    @property
    def describe(self) -> str:
        return f"reference(k={self.k}, embed={self.embedder.name}, model={self.model.name})"

    def index(self, corpus: Corpus) -> None:
        self.retriever.index(corpus)

    def answer(self, question: str) -> Answer:
        hits = self.retriever.search(question, self.k)
        context = "\n\n".join(h.text for h in hits)
        text = self.model.complete(PROMPT.format(context=context, question=question))
        return Answer(text=text.strip(), retrieved=[h.chunk_id for h in hits])
