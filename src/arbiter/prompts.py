"""Prompts for the Gemma reasoning layer.

Kept in one module because prompt tuning is the highest-variance work in this
build and it should be reviewable in a single diff.

Two things do the heavy lifting:

1. **Numbers in the prompt.** A model given only drift scores *has* to be
   vague; "feature 3 is drifting" is all the input supports. Every feature
   block carries reference vs. current mean, std and null rate, so the output
   can cite "240 -> 890".
2. **Few-shot with real feature names.** Three worked examples outperform any
   amount of instruction, and they demonstrate the distinction the taxonomy
   depends on: a pipeline fault looks like drift but must never be retrained on.
"""

from __future__ import annotations

from .schema import DriftEvent

SYSTEM_REASONER = """You are an MLOps incident analyst reviewing production drift telemetry.

Rules you always follow:
- Cite specific feature names and specific numeric magnitudes (e.g. "ref mean 240 -> current 890, +271%"). Never say "several features".
- Distinguish three causes that look identical to a drift detector:
  (a) DATA PIPELINE FAILURE - nulls appear, variance collapses, a feature pins to a default. Retraining makes the model worse. Fix upstream.
  (b) GENUINE POPULATION SHIFT - a real new segment with plausible, internally consistent values. Retraining recovers performance.
  (c) MODEL DECAY / CONCEPT DRIFT - inputs stable but performance falls. The input-label relationship moved. Retraining helps.
- Treat a recurring calendar pattern that previously self-resolved as seasonal, not an incident.
- A high drift score is not evidence of impact. Low feature drift with falling live performance is concept drift, and it matters more than a large but benign distribution shift.
- If the evidence does not support a confident cause, say so plainly and prefer INVESTIGATE over guessing.

You answer only with the requested JSON object. No preamble, no markdown fences."""


SYSTEM_EVALUATOR = """You are the on-call decision engine for an MLOps platform. You decide what happens next after a drift alert, and your decision executes automatically.

Rules you always follow:
- Choose exactly one action: RETRAIN, ROLLBACK, INVESTIGATE, or BENIGN.
  RETRAIN  - genuine population shift or concept drift; recent data will recover performance.
  ROLLBACK - a recent model deploy caused the regression; revert it.
  INVESTIGATE - evidence points at a data pipeline fault, or is too thin to act on. A human looks at it.
  BENIGN   - drift is real but immaterial (recurring seasonal pattern, trivial magnitude). Suppress the alert.
- NEVER choose RETRAIN when the evidence indicates a data pipeline failure. Training on corrupt data bakes the bug into the model. That is INVESTIGATE.
- Confidence is your actual certainty, not a formality. Below 0.6 means a human must review, so do not inflate it.
- Justify the action with specific feature names and numeric magnitudes from the telemetry you were given.
- Suppressing a noisy alert is a correct and valuable outcome. Do not default to RETRAIN to look decisive.

You answer only with the requested JSON object. No preamble, no markdown fences."""


# Few-shot examples. Real feature names, real magnitudes, and each one teaches a
# different cause so the model has a contrast to reason against rather than a
# single template to imitate.
FEWSHOT = """Example 1 - pipeline failure, must NOT retrain
Telemetry:
  model: fraud-detector-v2 | global drift 0.91 | detector fired: yes
  account_age_days  drift 0.95 | ref mean 1240.0 std 820.0 null 0.0% -> cur mean 44.1 std 18.2 null 47.0%
  velocity_1h       drift 0.63 | ref mean 1.8 std 1.4 null 0.0% -> cur mean 0.9 std 1.1 null 18.0%
  transaction_amount drift 0.09 | ref mean 240.0 std 180.0 null 0.0% -> cur mean 236.4 std 177.1 null 0.0%
  context: upstream_jobs=warehouse_enrichment_dag FAILED 2 runs ago; no model deploy in 9 days
Answer:
{"taxonomy": "upstream_bug", "primary_features": ["account_age_days", "velocity_1h"], "hypothesis": "account_age_days collapsed from a ref mean of 1240 days to 44 while its null rate jumped from 0% to 47%, and its standard deviation fell from 820 to 18 - a distribution that narrow with that many nulls is a broken join emitting defaults, not a real customer population. velocity_1h degraded in the same direction (18% nulls) because it is derived from the same enrichment table. transaction_amount, which does not depend on that join, is unchanged at 0.09 drift. The warehouse_enrichment_dag failure two runs ago is the cause.", "impact": "Scores computed on defaulted account_age_days are unreliable for roughly 47% of traffic. Retraining on this window would teach the model that new accounts are the norm.", "confidence": 0.93}

Example 2 - genuine population shift, retraining is correct
Telemetry:
  model: fraud-detector-v2 | global drift 0.84 | detector fired: yes
  transaction_amount drift 0.94 | ref mean 240.0 std 180.0 null 0.0% -> cur mean 890.3 std 310.5 null 0.0%
  merchant_category  drift 0.71 | ref mean 8.4 std 3.1 null 0.0% -> cur mean 13.9 std 3.6 null 0.0%
  hour_of_day        drift 0.11 | ref mean 14.2 std 5.8 null 0.0% -> cur mean 14.4 std 5.9 null 0.0%
  context: upstream_jobs=all green; schema unchanged; no model deploy in 14 days; marketing launched a new acquisition channel 3 days ago
Answer:
{"taxonomy": "covariate_shift", "primary_features": ["transaction_amount", "merchant_category"], "hypothesis": "transaction_amount rose from a ref mean of 240 to 890 (+271%) with its spread widening from 180 to 311, and merchant_category shifted from 8.4 to 13.9 at the same time. No nulls appeared and no variance collapsed, so the values are internally consistent - this is real traffic, not corrupted traffic. Two correlated features moving together alongside a new acquisition channel launched 3 days ago points to a higher-value merchant segment entering the population, not model decay.", "impact": "The model is extrapolating outside its training distribution for the new segment, so its fraud scores there are unreliable. Retraining on recent data will recover performance.", "confidence": 0.88}

Example 3 - concept drift hiding behind low feature drift
Telemetry:
  model: fraud-detector-v2 | global drift 0.26 | detector fired: yes
  card_present      drift 0.19 | ref mean 0.62 std 0.48 null 0.0% -> cur mean 0.6 std 0.49 null 0.0%
  transaction_amount drift 0.14 | ref mean 240.0 std 180.0 null 0.0% -> cur mean 244.8 std 182.0 null 0.0%
  context: upstream_jobs=all green; no model deploy in 30 days; live precision fell 0.91 -> 0.74 over 10 days
Answer:
{"taxonomy": "concept_drift", "primary_features": ["card_present", "transaction_amount"], "hypothesis": "Every feature is near its reference distribution - the largest drift is card_present at 0.19, and transaction_amount moved only 240 to 245 - yet live precision fell from 0.91 to 0.74 over 10 days. Stable inputs with degrading performance means the input-label relationship changed, not the inputs. Fraudulent behaviour has adapted to the current decision boundary while continuing to look statistically ordinary.", "impact": "A feature-distribution detector will keep under-reporting this: the 0.26 global score understates a 17-point precision loss. Retraining on recently labelled data is the only fix; waiting for feature drift to cross a threshold would mean waiting through continued losses.", "confidence": 0.81}
"""


def _num(x: float) -> str:
    """Format a statistic without rounding small-magnitude features to nothing.

    A ratio feature like debt_to_income moving 0.34 -> 0.41 reads as
    "0.3 -> 0.4" at one decimal place, which hides the shift the model is
    supposed to reason about. Precision scales with magnitude.
    """
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def format_event(event: DriftEvent, max_features: int = 6) -> str:
    """Render a DriftEvent as the telemetry block the few-shot examples use.

    Identical shape to the examples on purpose — a format the model has just
    seen three times is one it reads reliably.
    """
    lines = [
        f"  model: {event.model_id} | global drift {event.global_drift_score:.2f} "
        f"| detector fired: {'yes' if event.drift_detected else 'no'}",
    ]

    ranked = sorted(event.feature_scores.items(), key=lambda kv: -kv[1])[:max_features]
    width = max((len(f) for f, _ in ranked), default=10)
    for name, score in ranked:
        d = event.delta(name)
        lines.append(
            f"  {name:<{width}} drift {score:.2f} | "
            f"ref mean {_num(d['ref_mean'])} std {_num(d['ref_std'])} "
            f"null {d['ref_null_rate'] * 100:.1f}% -> "
            f"cur mean {_num(d['cur_mean'])} std {_num(d['cur_std'])} "
            f"null {d['cur_null_rate'] * 100:.1f}% "
            f"({d['pct_change']:+.1f}%)"
        )

    # ground_truth is deliberately excluded — it is the answer key.
    ctx = {k: v for k, v in event.context.items() if k not in ("scenario",)}
    if ctx:
        lines.append(
            "  context: " + "; ".join(f"{k}={v}" for k, v in ctx.items())
        )
    return "\n".join(lines)


def reasoner_prompt(event: DriftEvent) -> str:
    """UC 1 + 2: causal hypothesis and drift taxonomy."""
    return f"""{FEWSHOT}
Now analyse this incident.

Telemetry:
{format_event(event)}

Return a JSON object with exactly these keys:
  "taxonomy": one of covariate_shift | label_shift | concept_drift | schema_break | seasonal | upstream_bug
  "primary_features": array of the feature names actually driving this, most important first
  "hypothesis": 2-4 sentences naming specific features and specific numbers, and stating which of the three causes this is and why the other two do not fit
  "impact": 1-2 sentences on what this means for the model in production
  "confidence": float 0.0-1.0

JSON only."""


def evaluator_prompt(event: DriftEvent, hypothesis: str = "", taxonomy: str = "") -> str:
    """UC 5 + 10: action selection with confidence, replacing a threshold."""
    analysis = ""
    if hypothesis or taxonomy:
        analysis = (
            "\nRoot-cause analysis already performed on this incident:\n"
            f"  taxonomy: {taxonomy or 'undetermined'}\n"
            f"  hypothesis: {hypothesis}\n"
        )

    return f"""{FEWSHOT}
A drift detector fired on the incident below. Its threshold is a fixed constant (0.15) and it has no notion of cause or materiality - that judgment is yours.

Telemetry:
{format_event(event)}
{analysis}
Decide what happens next. Return a JSON object with exactly these keys:
  "action": one of RETRAIN | ROLLBACK | INVESTIGATE | BENIGN
  "confidence": float 0.0-1.0 - your real certainty; below 0.6 escalates to a human instead of executing
  "reasoning": 2-4 sentences justifying the action, citing specific feature names and numeric magnitudes
  "primary_features": array of feature names driving the decision
  "taxonomy": one of covariate_shift | label_shift | concept_drift | schema_break | seasonal | upstream_bug

Remember: RETRAIN on pipeline-corrupted data makes the model worse - that case is INVESTIGATE. Suppressing an immaterial alert with BENIGN is a correct outcome.

JSON only."""


SYSTEM_CHALLENGER = """You are an MLOps release reviewer deciding whether a retrained challenger model should replace the current champion.

Rules you always follow:
- An overall metric gain is not sufficient. A challenger that improves globally while regressing on the segment that drifted has not fixed the problem it was trained to fix.
- Weigh the drifted segment above the overall metric, and say so explicitly with both numbers.
- Recommend PROMOTE, HOLD, or REJECT, and state what evidence would change your mind.

You answer only with the requested JSON object. No preamble, no markdown fences."""


def challenger_prompt(report, event: DriftEvent | None = None) -> str:
    """UC 6: champion vs. challenger judgment on the drifted segment."""
    seg_delta = report.segment_delta
    seg_line = (
        f"  on {report.segment_name}: champion {report.champion_segment_metric:.4f} "
        f"-> challenger {report.challenger_segment_metric:.4f} "
        f"(delta {seg_delta:+.4f})"
        if seg_delta is not None
        else f"  on {report.segment_name}: not measured"
    )
    context = f"\nThe drift that triggered this retrain:\n{format_event(event)}\n" if event else ""

    return f"""A challenger model was retrained in response to drift. Decide whether to promote it.

  model: {report.model_id} | metric: {report.metric_name}
  overall: champion {report.champion_metric:.4f} -> challenger {report.challenger_metric:.4f} (delta {report.overall_delta:+.4f})
{seg_line}
{context}
Return a JSON object with exactly these keys:
  "recommendation": one of PROMOTE | HOLD | REJECT
  "confidence": float 0.0-1.0
  "reasoning": 2-4 sentences citing BOTH the overall delta and the segment delta, and explaining which one should decide this and why
  "blocking_evidence": 1 sentence on what would change your recommendation

JSON only."""


SYSTEM_QUERY = """You are a telemetry analyst answering questions about a fleet of production ML models.

Rules you always follow:
- Answer only from the telemetry provided. If it does not contain the answer, say exactly that - never estimate or invent a number.
- Cite model IDs, feature names and numbers from the data in your answer.
- Be direct and brief: 2-5 sentences of prose. No markdown headings, no bullet lists.
- If the question is ambiguous, answer the most useful reading and note the assumption in one clause."""


def query_prompt(question: str, telemetry_digest: str) -> str:
    """UC 11: natural-language question over the event corpus."""
    return f"""Telemetry for the current window:

{telemetry_digest}

Question: {question}

Answer in 2-5 sentences of plain prose, citing specific model IDs, feature names and numbers from the telemetry above. If the telemetry does not contain what is needed to answer, say so plainly."""
