"""Console and HTML reporting."""
from __future__ import annotations

import html
from pathlib import Path

from .operators import WEIGHT, Kind
from .outcomes import Outcome, Run


def console(run: Run) -> str:
    c = run.calibration
    L = ["mirage - evidence sensitivity report", ""]
    L.append(f"pipeline: {run.pipeline}")
    L.append(f"model:    {run.model}")
    L.append("")
    L.append(f"baseline answer variance   mean self-similarity {c.mean_self_similarity:.3f} "
             f"(k={c.k}, sd={c.stdev:.3f})")
    L.append(f"difference threshold       {c.threshold:.3f}   [{c.source}]")
    if c.paraphrase_floor is not None:
        L.append(f"paraphrase floor           {c.paraphrase_floor:.3f}   "
                 f"(an answer must move further than a paraphrase does)")
    L.append("")

    ctrl = run.controls()
    if ctrl:
        L.append("CONTROLS - the answer SHOULD hold for these")
        for kind, (held, total) in sorted(ctrl.items(), key=lambda x: x[0].value):
            pct = 100 * held / total if total else 0.0
            L.append(f"  {kind.value:24} {held}/{total} held  ({pct:.0f}%)")
        L.append(f"  verdict                  {'PASS' if run.controls_pass() else 'FAIL'}")
        L.append("")

    if run.inconclusive():
        L.append("INCONCLUSIVE - no score is reported.")
        L.append("  Either no mutant was scored, or the controls failed. A control failure")
        L.append("  means the pipeline changes its answer for reasons unrelated to the")
        L.append("  evidence, so a sensitivity score would not mean anything.")
        return "\n".join(L)

    surv = run.survivors()
    if surv:
        L.append("SURVIVORS - evidence corrupted, answer unchanged. Highest weight first.")
        for r in surv:
            L.append(f"  {r.mutant.id}  w{WEIGHT[r.mutant.kind]:.2f}  {r.mutant.kind.value:20} "
                     f"sim {r.similarity:.3f}")
            L.append(f"        Q: {r.mutant.question}")
            L.append(f"        A: {r.baseline_answer[:100]}")
            L.append(f"        {r.mutant.description}")
        L.append("")

    L.append("BY OPERATOR")
    for kind, (killed, total) in sorted(run.by_operator().items(), key=lambda x: -WEIGHT[x[0]]):
        if kind in (Kind.PARAPHRASE, Kind.IRRELEVANT):
            continue
        pct = 100 * killed / total if total else 0.0
        L.append(f"  {kind.value:24} {killed:>3}/{total:<3} {pct:5.1f}%   weight {WEIGHT[kind]:.2f}")
    L.append("")

    scored = run.scored()
    killed = sum(1 for r in scored if r.outcome is Outcome.KILLED)
    nv = sum(1 for r in run.results if r.outcome is Outcome.NOT_VIABLE)
    err = sum(1 for r in run.results if r.outcome is Outcome.ERROR)
    L.append("METRICS")
    L.append(f"  planned mutants        {len(run.results)}")
    L.append(f"  not viable             {nv}  (excluded)")
    L.append(f"  errors                 {err}  (excluded, never hidden)")
    L.append(f"  scored                 {len(scored)}")
    L.append(f"  killed                 {killed}")
    L.append(f"  survived               {len(scored) - killed}")
    L.append(f"  EVIDENCE SENSITIVITY   {100 * run.sensitivity():.1f}%   (killed / scored)")
    L.append(f"  weighted sensitivity   {100 * run.weighted_sensitivity():.1f}%   "
             f"(deletion counts most)")
    L.append(f"  embedding cache        {run.embed_hits} reuses, {run.embed_misses} computed")
    L.append(f"  wall time              {run.wall_ms / 1000:.1f}s")
    return "\n".join(L)


def write_html(run: Run, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in run.results:
        colour = {"KILLED": "#0b7a55", "SURVIVED": "#b3261e"}.get(r.outcome.value, "#666")
        rows.append(
            f"<tr><td>{r.mutant.id}</td><td>{r.mutant.kind.value}</td>"
            f"<td style='color:{colour};font-weight:600'>{r.outcome.value}</td>"
            f"<td>{r.similarity:.3f}</td><td>{html.escape(r.mutant.question)}</td>"
            f"<td>{html.escape(r.baseline_answer[:120])}</td>"
            f"<td>{html.escape(r.mutated_answer[:120])}</td></tr>")
    verdict = ("INCONCLUSIVE" if run.inconclusive()
               else f"{100 * run.sensitivity():.1f}% evidence sensitivity")
    path.write_text(f"""<!doctype html><meta charset=utf-8>
<title>mirage report</title>
<style>body{{font:14px system-ui;margin:32px;color:#1a1a1a}}
h1{{color:#0e5c54}} table{{border-collapse:collapse;width:100%;font-size:12px}}
td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
.big{{font-size:28px;font-weight:700;color:#0e5c54}}</style>
<h1>mirage — evidence sensitivity</h1>
<p class=big>{verdict}</p>
<p>pipeline: <code>{html.escape(run.pipeline)}</code><br>
model: <code>{html.escape(run.model)}</code><br>
difference threshold: {run.calibration.threshold:.3f}
(baseline self-similarity {run.calibration.mean_self_similarity:.3f}, k={run.calibration.k})</p>
<table><tr><th>id</th><th>operator</th><th>outcome</th><th>sim</th>
<th>question</th><th>baseline answer</th><th>mutated answer</th></tr>
{''.join(rows)}</table>""", encoding="utf-8")
