# AI agents in Arbiter

Six distinct Gemma 4 agents, each with its own system prompt, its own context,
and its own output contract. They are separate because the questions are
separate: an agent asked to find a root cause reasons differently from one
asked whether to execute an action, and merging them produces worse answers to
both.

All six run on `gemma-4-26b-a4b-it` (open-weight, hosted inference) at
`temperature=0.2` with a pinned model ID.

---

## 1 · Reasoner — "why did this happen?"

| | |
|---|---|
| **Module** | [`reasoner.py`](../src/arbiter/reasoner.py) · `analyze()` |
| **Prompt** | `prompts.SYSTEM_REASONER` + three few-shot examples |
| **Context** | One `DriftEvent`: per-feature drift scores, reference vs. current mean/std/null-rate, and free-form context (upstream job status, deploy markers) |
| **Returns** | `Analysis` — taxonomy, primary features, causal hypothesis, production impact, confidence |
| **Fallback** | Deterministic heuristic, marked `source="heuristic"` |

Its whole job is one distinction: **a data-pipeline failure, a genuine
population shift, and model decay produce similar drift scores but demand
opposite responses.** The system prompt names the signatures — nulls appearing
and variance collapsing toward a default means a broken join; internally
consistent values with no nulls means real traffic; stable inputs with falling
performance means the input-label relationship moved.

Three few-shot examples carry more weight here than any amount of instruction,
and each teaches a different cause so the model has a contrast rather than a
template.

## 2 · Evaluator — "does it matter, and what do we do?"

| | |
|---|---|
| **Module** | [`evaluator.py`](../src/arbiter/evaluator.py) · `decide()` |
| **Prompt** | `prompts.SYSTEM_EVALUATOR` |
| **Context** | The same `DriftEvent`, **plus the Reasoner's hypothesis and taxonomy** |
| **Returns** | `Verdict` — `RETRAIN` / `ROLLBACK` / `INVESTIGATE` / `BENIGN`, confidence, reasoning, primary features |
| **Fallback** | Taxonomy-driven heuristic that never retrains a pipeline fault |

Runs *after* the Reasoner by default. A decision made without a cause is the
threshold problem again in a different shape.

The prompt states explicitly that suppressing an immaterial alert is a correct
and valuable outcome — without that, the model defaults to looking decisive and
recommends retraining on everything.

## 3 · Challenger reviewer — "should this retrained model be promoted?"

| | |
|---|---|
| **Module** | [`evaluator.py`](../src/arbiter/evaluator.py) · `judge_challenger()` |
| **Prompt** | `prompts.SYSTEM_CHALLENGER` |
| **Context** | Champion vs. challenger metrics, overall **and** on the drifted segment |
| **Returns** | `PROMOTE` / `HOLD` / `REJECT` with reasoning and blocking evidence |

Exists because the overall metric is the number that misleads. A challenger can
gain globally while regressing on exactly the segment it was retrained for, and
the prompt requires both numbers to be cited.

## 4 · Model chat — per-model questions

| | |
|---|---|
| **Module** | [`chat.py`](../src/arbiter/chat.py) · `ask_model()` |
| **Prompt** | `chat.SYSTEM_MODEL_CHAT` |
| **Context** | One model's feature names, recent window statistics, drift-score summary, and up to 10 reasoned incidents — plus the last 6 conversation turns |
| **Returns** | 2–6 sentences of prose |

Scoped to a single model so it can answer with real magnitudes. Instructed to
say plainly when the context does not contain the answer rather than estimating.

## 5 · Fleet chat — cross-model questions

| | |
|---|---|
| **Module** | [`chat.py`](../src/arbiter/chat.py) · `ask_fleet()` |
| **Prompt** | `chat.SYSTEM_FLEET_CHAT` |
| **Context** | One summary line per model the account owns, plus recent verdicts |
| **Returns** | 2–6 sentences of prose |

Deliberately given *less* per-model detail. The prompt tells it to answer what
the summary supports and point the user at a model's own chat for
feature-level analysis, rather than inventing numbers it was not given.

## 6 · Doc writer — incident documentation

| | |
|---|---|
| **Module** | [`docgen.py`](../src/arbiter/docgen.py) · `generate()` |
| **Prompt** | `docgen.SYSTEM_DOCS` |
| **Context** | A factual incident log assembled server-side: model metadata, drift-score summary, and every incident with per-feature movement |
| **Returns** | Markdown — overview, incident history, root-cause patterns, recommended actions, monitoring notes |
| **Fallback** | The raw log, clearly labelled as lacking analysis |

The incident list is built in code and handed to the model, so the report
**cannot contain an incident that did not happen**. Gemma writes the prose; it
does not choose the facts.

---

## Safety rails

These apply to every agent that can cause an action.

### Pipeline faults can never trigger a retrain

A prompt instruction is not a control, so the rule is enforced in code:

```python
# evaluator.py
if verdict.action == "RETRAIN" and verdict.taxonomy in ("upstream_bug", "schema_break"):
    verdict.action = "INVESTIGATE"      # training on corrupt data bakes in the bug
    verdict.confidence = min(verdict.confidence, CONFIDENCE_FLOOR - 0.01)
```

Tested with a mocked response demanding `RETRAIN` at 0.95 confidence
(`tests/test_evaluator.py::TestPipelineFaultGuard`).

### Low confidence escalates instead of executing

Below `CONFIDENCE_FLOOR = 0.60` a verdict is routed to a human. This is what
makes automated action defensible at all.

### Hallucinated feature names are dropped

Any feature the model names that is not in the event is removed before the
verdict is stored. The UI renders these as fact, so an invented name would be
worse than an empty list.

### Malformed JSON never crashes a verdict

Gemma has no guaranteed JSON mode on this API, so `gemma_client.repair_json`
strips fences, extracts an object from surrounding prose, and repairs trailing
commas. When nothing object-shaped can be recovered it returns `None` and the
caller falls back — it never invents a verdict.

### Fallback results are not reportable

Every verdict records whether it came from Gemma or the offline heuristic.
`measure_suppression` returns `reportable: false` and `measure_accuracy`
returns `None` unless every verdict in the run was Gemma-backed — the heuristic
derives its taxonomy from the same signals the test fixtures were built from,
so grading it against those labels scores the fixture generator against itself.

### Key rotation keeps reasoning available

[`keypool.py`](../src/arbiter/keypool.py) spreads calls across every configured
key, round robin. A rate-limited key is parked for 60 seconds and the next key
takes over, so one exhausted quota does not degrade the whole dashboard to
fallback output. See [operations.md](./operations.md#api-key-rotation).
