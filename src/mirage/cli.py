"""mirage command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import Corpus
from .engine import run as run_engine
from .llm import ReplayModel
from .pipeline import ReferencePipeline
from .report import console, write_html


def _load(project: Path) -> tuple[dict[str, str], list[str]]:
    docs = json.loads((project / "corpus.json").read_text(encoding="utf-8"))
    qs = json.loads((project / "questions.json").read_text(encoding="utf-8"))
    return docs, qs


def _pipeline(args):
    parametric: dict[str, str] = {}
    p = Path(args.project) / "parametric.json"
    if args.ungrounded and p.exists():
        parametric = json.loads(p.read_text(encoding="utf-8"))
    if args.model == "anthropic":
        from .llm import AnthropicModel
        model = AnthropicModel()
    else:
        model = ReplayModel(parametric)
    if args.pipeline == "langgraph":
        # Raises if the optional extra is absent. Never falls back: the report
        # names the pipeline it measured, so it must have measured that one.
        from .adapters.langgraph_pipeline import LangGraphPipeline
        return LangGraphPipeline(model=model)
    return ReferencePipeline(model=model)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mirage")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("scan", "test"):
        s = sub.add_parser(name)
        s.add_argument("project")
        s.add_argument("--model", default="replay", choices=["replay", "anthropic"])
        s.add_argument("--pipeline", default="reference",
                       choices=["reference", "langgraph"],
                       help="which RagPipeline implementation to measure")
        s.add_argument("--k", type=int, default=5, help="calibration runs per question")
        s.add_argument("--ungrounded", action="store_true",
                       help="use the parametric-memory model — the false-confidence demo")
        s.add_argument("--exhaustive", action="store_true",
                       help="disable coverage-guided selection (for the A/B)")
        s.add_argument("--out", default="mirage-out")
        if name == "test":
            s.add_argument("--threshold", type=float, default=0.90)

    args = ap.parse_args(argv)
    project = Path(args.project)
    docs, questions = _load(project)
    corpus = Corpus.from_documents(docs)
    pipeline = _pipeline(args)

    result = run_engine(pipeline, corpus, questions, k=args.k,
                        coverage_guided=not args.exhaustive)
    print(console(result))
    out = Path(args.out)
    write_html(result, out / "report.html")
    print(f"\n  html                   {out / 'report.html'}")

    if args.cmd == "test":
        if result.inconclusive():
            print("\nGATE: INCONCLUSIVE — controls failed or nothing was scored.")
            return 2
        score = result.sensitivity()
        ok = score >= args.threshold
        print(f"\nGATE: {'PASS' if ok else 'FAIL'}   "
              f"{100 * score:.1f}% vs threshold {100 * args.threshold:.0f}%")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
