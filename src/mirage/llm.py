"""Model access behind one protocol.

ReplayModel is what makes mirage runnable with no API key and no network — the
same decision mutagent made, and the reason a demo can be given offline. Every
report prints which model produced it, so a replay run can never be mistaken
for a live one.
"""
from __future__ import annotations

import os
import re
from typing import Protocol


class ChatModel(Protocol):
    def complete(self, prompt: str) -> str: ...
    @property
    def name(self) -> str: ...


class ReplayModel:
    """A deterministic stand-in for a grounded reader.

    It answers ONLY from the context it is given, by extracting the span that
    answers the question. That makes it a *faithful* reader by construction —
    which is exactly what mirage needs as its control: a pipeline that SHOULD
    score highly, so a low score means the tool found something real.
    """

    def __init__(self, parametric: dict[str, str] | None = None):
        # Facts the model "remembers" regardless of context — how a real LLM
        # answers from pretraining instead of from retrieval. This is what
        # creates survivors, and it is how the false-confidence fixture works.
        self.parametric = parametric or {}

    @property
    def name(self) -> str:
        return "replay (deterministic reader, not model output)"

    def complete(self, prompt: str) -> str:
        question = _section(prompt, "QUESTION")
        context = _section(prompt, "CONTEXT")
        for key, answer in self.parametric.items():
            if key.lower() in question.lower():
                return answer
        if not context.strip():
            return "I don't know."
        best, best_score = "", 0.0
        qt = set(re.findall(r"[a-z0-9]+", question.lower()))
        for sentence in re.split(r"(?<=[.!?])\s+", context):
            st = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            if not st:
                continue
            score = len(qt & st) / (len(qt) ** 0.5 + 1e-9)
            if score > best_score:
                best, best_score = sentence.strip(), score
        return best if best_score > 0.5 else "I don't know."


class AnthropicModel:
    """Live path. UNVERIFIED: never exercised against the API in this repo.

    The key is read from the environment only and is never written to a report,
    trace or checkpoint.
    """

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 512):
        self.model, self.max_tokens = model, max_tokens
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, prompt: str) -> str:  # pragma: no cover - unverified
        import json
        import urllib.request
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
        return "".join(b.get("text", "") for b in body.get("content", []))


def _section(prompt: str, header: str) -> str:
    m = re.search(rf"^{header}:\s*\n(.*?)(?=\n[A-Z]+:|\Z)", prompt, re.S | re.M)
    return m.group(1) if m else ""
