# Module reference

Everything in `src/arbiter/`. Line counts are from the working tree.

## Core contract

### `schema.py` — 231 lines
The data contract every other module speaks: `DriftEvent`, `Verdict`,
`ChallengerReport`.

Holds **`map_feature_scores`**, the highest-value function in the codebase. The
SDK's `ADWINDriftDetector.get_status()` returns feature scores as a positional
list because the detector only sees a matrix. Gemma cannot say
"transaction_amount drifted" from `[0.56, 0.27, …]`, so this maps indices to the
ordered names captured at registration. Accepts list, tuple and mapping shapes;
an out-of-range index becomes `feature_<i>` rather than being dropped silently.

`Verdict.__post_init__` coerces defensively: an unrecognised action becomes
`INVESTIGATE` (escalate, never act on a parse failure), confidence is clamped to
0–1, and an unknown taxonomy becomes `None`.

### `prompts.py` — 236 lines
Every system prompt and the three few-shot examples, in one module because
prompt tuning is the highest-variance work in the build and should be reviewable
in a single diff. `format_event` renders telemetry in the same shape as the
examples, with adaptive decimal precision — a ratio feature printed as
`0.3 -> 0.3` hides the shift it is supposed to explain.

## Reasoning

### `reasoner.py` — 201 lines
UC 1 + 2. `analyze()` produces the causal hypothesis and taxonomy;
`meets_quality_bar()` checks a narration names a specific feature, cites a
magnitude, assigns a taxonomy and distinguishes cause — measured, not assumed.

### `evaluator.py` — 371 lines
UC 5, 6, 10. `decide()` selects the action, `judge_challenger()` reviews a
retrained model, `gate()` is the `@dg.retrainer` seam, and
`measure_suppression()` / `measure_accuracy()` produce the README's numbers —
refusing to report when any verdict came from the fallback.

### `chat.py` — 338 lines
Two chatbots. `ask_model()` gets one model's features, window statistics and
verdicts; `ask_fleet()` gets one summary line per model. The context difference
is the design.

### `docgen.py` — 245 lines
Gemma-written incident reports. `build_incident_log()` assembles the facts in
code and `generate()` asks Gemma to narrate them, so a report cannot contain an
incident that did not happen.

## Infrastructure

### `gemma_client.py` — 281 lines
Hosted Gemma with three failure modes handled: no guaranteed JSON mode
(`repair_json`), the network (every response cached to `demo_cache.json` for
offline replay), and a missing key (serves cache, reports `available == False`).
Calls route through the key pool and retry on the next key when rate limited.

### `keypool.py` — 196 lines
Round-robin pool over `GEMMA_API_KEY` and `GEMMA_API_KEY_2..5`. A 429 parks the
key for 60 seconds; the next call uses a different one. `stats()` reports pool
health and contains no key material.

### `keys.py` — 479 lines
SQLite store and API key issuance. Keys are salted SHA-256 — the plaintext is
returned once at creation and is unrecoverable. All reads are user-scoped.

### `auth.py` — 279 lines
Accounts and sessions. PBKDF2-HMAC-SHA256, 600k iterations, per-user salt,
constant-time compare. Email changes require the current password; a password
change invalidates every session.

### `ingest.py` — 466 lines
The SDK wire protocol, read off `driftguard-ai-sdk` 1.0.4 rather than guessed.
Stores telemetry on the hot path (~40ms) and hands reasoning to a daemon thread.
Contains the window-selection logic described in
[architecture.md](./architecture.md#window-selection).

### `api.py` — 676 lines
FastAPI layer. See [api.md](./api.md).

### `env.py` — 34 lines
Minimal `.env` loader, so no dependency is added for one small job. Real
environment variables always win.

## Demo and verification

### `scenarios.py` — 322 lines
Four synthetic drift families with hand-labelled causes, fixed seed. Every event
fires the detector — that is the point, since the suppression count is measured
against alerts a threshold would have raised. `ground_truth` is never shown to
the model.

### `run_replay.py` — 127 lines
CLI producing the README's measured numbers. Prints `UNREPORTABLE RUN` when any
verdict came from the fallback.

### `sdk_demo.py` — 255 lines
Live proof: trains a real champion model, runs the real detector, maps the
scores, measures the real accuracy cost, and sends it to Gemma.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/e2e_flow.py` | Full flow in-process: account, key, 3 models, drift, verdicts, isolation, settings |
| `scripts/seed_models.py` | The same against a running server over HTTP |

## Tests

| File | Covers |
|---|---|
| `tests/test_schema.py` | The contract, hardest on index-to-name mapping |
| `tests/test_gemma_client.py` | JSON repair, cache replay, degraded mode |
| `tests/test_evaluator.py` | Pipeline-fault guard, confidence escalation, measurement honesty |
| `tests/test_ingest.py` | Key issuance, auth enforcement, SDK protocol, hot-path latency |
| `tests/test_keypool.py` | Rotation, rate-limit failover, no key material in stats |
