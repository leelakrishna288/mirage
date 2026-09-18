"""The same RAG pipeline, expressed as a LangGraph state graph.

Why this exists
---------------
mirage must not be "a tool for pipelines I happen to have written". The
``RagPipeline`` protocol is the claim; this adapter is the evidence. It runs the
identical retrieval and prompting steps as ``ReferencePipeline``, but as an
explicit graph — retrieve -> build context -> generate — with typed state
flowing between nodes.

``test_langgraph_equivalence`` asserts the two produce the SAME answers on the
fixture. That is the assertion worth having: if the graph and the reference ever
diverge, a measurement taken through one would not describe the other.

LangGraph is an OPTIONAL extra (``pip install -e ".[langgraph]"``). mirage's two
hard dependencies stay numpy and scikit-learn, and the whole tool still runs with
no network. Asking for this adapter without the extra installed RAISES rather
than silently falling back — a run must never quietly measure a different
pipeline from the one it reports.
"""
from __future__ import annotations

from typing import Any, TypedDict

from ..corpus import Corpus
from ..embed import CachedEmbedder, HashingEmbedder
from ..llm import ChatModel, ReplayModel
from ..pipeline import PROMPT, Answer
from ..retrieve import HybridRetriever


class GraphState(TypedDict):
    """State passed between nodes. Explicit, so the graph is readable."""
    question: str
    retrieved: list[str]
    context: str
    answer: str


def _require_langgraph() -> Any:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as e:  # pragma: no cover - exercised by the skip path
        raise RuntimeError(
            "The LangGraph adapter needs the optional extra: "
            'pip install -e ".[langgraph]". Refusing to fall back to the '
            "reference pipeline — a run must measure the pipeline it names."
        ) from e
    return START, END, StateGraph


class LangGraphPipeline:
    """A RagPipeline built as a compiled LangGraph StateGraph."""

    def __init__(self, model: ChatModel | None = None, k: int = 4):
        START, END, StateGraph = _require_langgraph()
        self.embedder = CachedEmbedder(HashingEmbedder())
        self.retriever = HybridRetriever(self.embedder, k=k)
        self.model = model or ReplayModel()
        self.k = k

        g = StateGraph(GraphState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("build_context", self._build_context)
        g.add_node("generate", self._generate)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "build_context")
        g.add_edge("build_context", "generate")
        g.add_edge("generate", END)
        self._graph = g.compile()

    # -- nodes ------------------------------------------------------------
    def _retrieve(self, state: GraphState) -> dict[str, Any]:
        hits = self.retriever.search(state["question"], self.k)
        return {"retrieved": [h.chunk_id for h in hits],
                "context": "\n\n".join(h.text for h in hits)}

    def _build_context(self, state: GraphState) -> dict[str, Any]:
        # Separate node on purpose: prompt construction is a step worth seeing
        # in the graph, and the step most likely to change in a real system.
        return {"context": state["context"]}

    def _generate(self, state: GraphState) -> dict[str, Any]:
        text = self.model.complete(
            PROMPT.format(context=state["context"], question=state["question"]))
        return {"answer": text.strip()}

    # -- RagPipeline ------------------------------------------------------
    @property
    def describe(self) -> str:
        return (f"langgraph(k={self.k}, embed={self.embedder.name}, "
                f"model={self.model.name})")

    def index(self, corpus: Corpus) -> None:
        self.retriever.index(corpus)

    def answer(self, question: str) -> Answer:
        out = self._graph.invoke(
            {"question": question, "retrieved": [], "context": "", "answer": ""})
        return Answer(text=out["answer"], retrieved=out["retrieved"])
