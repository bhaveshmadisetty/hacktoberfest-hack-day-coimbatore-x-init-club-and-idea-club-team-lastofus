# Arbiter — documentation

Reference material for people working on Arbiter. The project pitch, measured
results and setup instructions live in the [top-level README](../README.md);
this folder covers how the system is put together.

| Document | What it covers |
|---|---|
| [architecture.md](./architecture.md) | How a prediction becomes a verdict: request flow, storage, the reasoning worker |
| [agents.md](./agents.md) | Every AI agent in the system — what each one is asked, what context it gets, what it returns |
| [modules.md](./modules.md) | Module-by-module reference for `src/arbiter/` |
| [api.md](./api.md) | HTTP reference: auth, dashboard API, and the SDK wire protocol |
| [operations.md](./operations.md) | Running it: environment, key rotation, failure modes, verification |

## The shape of the system in one paragraph

Your model sends per-prediction telemetry to Arbiter using the drift SDK's
existing wire protocol — two environment variables, no code change. Arbiter
stores it, and when drift crosses the model's threshold *and* enough drifted
samples have accumulated, a background worker builds a `DriftEvent` from the
telemetry (reference window vs. drifted window, per feature) and asks Gemma 4
two questions: what caused this, and what should happen next. The verdict is
stored and surfaced in the dashboard, where every number is paired with the
sentence that explains it.

## The two ideas worth knowing

**Positional scores are useless until they are named.** The drift detector
reports feature scores by position (`[0.56, 0.27, …]`) because it only ever
sees a matrix. Arbiter captures the ordered feature names at registration and
maps them before anything reaches the model — without that step the best
possible narration is "feature 0 drifted". See
[`schema.map_feature_scores`](./modules.md#schemapy).

**A verdict that cannot be trusted is not automated.** Verdicts below a 0.60
confidence floor escalate to a human instead of executing, and a `RETRAIN`
decision on a pipeline-fault taxonomy is blocked in code — a prompt
instruction is not a control. See [agents.md](./agents.md#safety-rails).
