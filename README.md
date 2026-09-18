# mirage

**Evidence-sensitivity testing for RAG systems.** Corrupt the evidence behind an answer, re-run the
pipeline, and measure whether the answer notices.

A citation is generated text — a model can cite a chunk it never read. mirage tests the causal link
directly: an answer that survives having its evidence deleted was not sensitive to that evidence.

```
$ mirage scan fixtures/false_confidence --ungrounded

baseline answer variance   mean self-similarity 1.000 (k=5, sd=0.000)
difference threshold       0.893   [self-consistency + paraphrase control]
paraphrase floor           0.913   (an answer must move further than a paraphrase does)

CONTROLS - the answer SHOULD hold for these
  irrelevant_corruption    4/4 held  (100%)
  paraphrase               4/4 held  (100%)
  verdict                  PASS

SURVIVORS - evidence corrupted, answer unchanged. Highest weight first.
  M001  w1.00  evidence_deletion    sim 1.000
        Q: What was Northwind Logistics total revenue in fiscal year 2025?
        A: Northwind Logistics reported total revenue of 412 million dollars...
        remove all 1 evidence chunk(s) from the index

BY OPERATOR
  evidence_deletion          2/4    50.0%   weight 1.00
  fact_corruption            2/4    50.0%   weight 0.90
  entity_swap                2/4    50.0%   weight 0.85
  contradiction              0/4     0.0%   weight 0.60
  distractor                 0/4     0.0%   weight 0.40

  EVIDENCE SENSITIVITY   30.0%
```

The document was deleted from the index. The system answered anyway. It was never reading it.

Measured on the same corpus and questions, verified on Linux (Python 3.11) and Windows (Python 3.14):

| Pipeline | Sensitivity | Weighted | Controls | `mirage test` |
|---|---|---|---|---|
| grounded (default) | 60.0% | 73.3% | PASS | exit 0 |
| ungrounded (`--ungrounded`) | 30.0% | 36.7% | PASS | exit 1 |

The grounded pipeline loses its 40% on the contradiction and distractor operators only — it is fully
sensitive to deletion, fact corruption and entity swaps. That split is the diagnosis.

## Why this and not the usual checks

| Approach | Why it is not enough |
|---|---|
| "It cites a source" | Citations are generated text. A model can cite a chunk it ignored. |
| LLM-as-judge | A model's opinion about a model's output. Fails in the same direction. |
| Retrieval recall@k | Measures whether the right chunk was *fetched*, not whether the answer *used* it. |

## The experimental design

**1. Measure the pipeline's own noise first.** The unmutated pipeline is run *k* times per question
and the answer-similarity distribution is recorded. Nothing downstream is interpretable without it:
if the pipeline already wobbles by 0.3, a mutant that moves the answer by 0.2 has told you nothing.

**2. Then let the paraphrase control set the bar.** Self-consistency only catches sampling noise. It
says nothing about legitimate *rewording*. An answer counts as changed only if it moves further than
a paraphrase of its own evidence moves it. (The first version skipped this and failed its own
control at 50% — see `test_refine_lowers_the_threshold_below_the_paraphrase_floor`.)

**3. Two controls, not one.**

| Control | Mutation | Expected |
|---|---|---|
| Positive — paraphrase | restate the same fact | answer holds |
| Negative — irrelevant | corrupt an off-topic chunk | answer holds |

If a control fails the run reports **INCONCLUSIVE** rather than a score. A pipeline that changes its
answer for reasons unrelated to the evidence cannot be scored for evidence sensitivity.

**4. Evidence sets, not single chunks.** If Document B independently states the same fact, mutating
only Document A produces a false survivor. mirage resolves every chunk carrying the claim and mutates
the set together.

## Operators

| Operator | Weight | A survivor suggests |
|---|---|---|
| Evidence deletion | 1.00 | there is no document the answer could have come from |
| Fact corruption | 0.90 | the figure did not come from the corpus |
| Entity swap | 0.85 | the subject came from parametric memory |
| Contradiction | 0.60 | conflicting evidence is ignored |
| Distractor | 0.40 | retrieval is not discriminating |
| Paraphrase | control | — |
| Irrelevant corruption | control | — |

## Framework-agnostic

Anything that can answer a question and name the chunks it retrieved can be measured:

```python
class RagPipeline(Protocol):
    def index(self, corpus: Corpus) -> None: ...
    def answer(self, question: str) -> Answer: ...   # text + retrieved chunk ids
```

Two implementations ship in-repo, and **CI asserts they agree**:

| Pipeline | How it is built | `--pipeline` |
|---|---|---|
| reference | plain Python, no external stack | `reference` (default) |
| LangGraph | a compiled `StateGraph`: retrieve → build context → generate | `langgraph` |

```bash
pip install -e ".[langgraph]"
mirage scan fixtures/false_confidence --pipeline langgraph
```

Both answer the same questions with the same text, the same retrieved chunks and
the same sensitivity score — `test_graph_answers_identically_to_the_reference_pipeline`
and a CI job pin it. If they ever diverge, a measurement taken through one would
not describe the other.

LangGraph is an **optional extra**: mirage's hard dependencies stay numpy and
scikit-learn, and the tool still runs offline. Asking for the adapter without the
extra installed **raises** rather than falling back — a report must name the
pipeline it actually measured.

## Install and run

```bash
pip install -e ".[dev]"
pytest -q                                             # 28 tests (33 with the extra)
mirage scan fixtures/false_confidence                 # grounded
mirage scan fixtures/false_confidence --ungrounded    # the false-confidence demo
mirage test fixtures/false_confidence --threshold 0.60
```

No API key and no network are needed: the default model is a deterministic replay reader, and every
report prints which model produced it so a replay run cannot be mistaken for a live one.

## CI gate

```yaml
# mirage.yaml
sensitivity_threshold: 0.90
```

`mirage test` exits non-zero when sensitivity falls below the threshold, and exits 2 when the run is
inconclusive. CI runs the gate twice: the grounded pipeline must pass and the ungrounded one must
fail — if the ungrounded pipeline ever passes, the tool has stopped working.

## What is honest about this repo

- **Not novel.** Perturbation and counterfactual grounding evaluation exist in the literature, and
  tools such as RAGAS measure adjacent things. The claim here is narrower: evidence perturbation
  turned into an automated mutation-testing framework with an operator taxonomy, two control groups,
  noise calibration and a CI gate.
- **A survivor is strong evidence, not a verdict.** The model may have reached the same answer via a
  chunk outside the resolved evidence set. Deletion survivors are strongest; distractor survivors
  weakest — which is what the weights encode.
- **The reference embedder is a hashing vectoriser**, not a trained sentence encoder. It keeps the
  demo offline and deterministic. Swap in a real encoder behind `Embedder` for production use.
- **The Anthropic adapter has never made a live call** and is marked unverified in its own docstring.
- **The reference pipeline scores 0% on contradiction and distractor operators.** That is a real,
  reported weakness of the reference reader, published rather than tuned away.
- **Cost:** every mutant is a model call. The offline replay model exists so the method can be
  demonstrated at zero cost.

## Layout

```
src/mirage/
  corpus.py      documents, chunks, snapshot/restore
  embed.py       Embedder protocol + content-hash cache
  retrieve.py    BM25 + dense, fused with RRF
  llm.py         ChatModel protocol, ReplayModel, AnthropicModel (unverified)
  pipeline.py    RagPipeline protocol + reference implementation
  evidence.py    evidence-set resolution
  operators.py   seven operators, two of them controls
  calibrate.py   noise calibration + paraphrase refinement
  compare.py     the answer-comparison ladder
  engine.py      baseline -> calibrate -> mutate -> re-run -> score
  outcomes.py    outcomes, sensitivity, per-operator breakdown
  report.py      console + HTML
  adapters/      alternative RagPipeline implementations (LangGraph)
  cli.py         mirage scan | mirage test
```

MIT licensed.
