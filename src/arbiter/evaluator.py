"""UC 5, 6, 10 — the judgment layer that replaces the threshold.

DriftGuard decides whether to retrain by comparing a drift score against a
fixed ``drift_threshold=0.15``. That constant cannot distinguish a broken
upstream join from a genuine new customer segment, and it has no notion of
whether acting would help. This module makes that call with reasoning instead:

- ``decide``            UC 5 + 10: action plus calibrated confidence
- ``judge_challenger``  UC 6: promote only if the drifted segment improved
- ``gate``              the ``@dg.retrainer`` seam — suppress or proceed

Low confidence escalates to a human rather than executing, which is what makes
automated action defensible at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .gemma_client import GemmaClient, get_client
from .prompts import (
    SYSTEM_CHALLENGER,
    SYSTEM_EVALUATOR,
    challenger_prompt,
    evaluator_prompt,
)
from .reasoner import Analysis, _heuristic, analyze
from .schema import ChallengerReport, DriftEvent, Verdict

# Below this, an action is not executed automatically — a human reviews it.
# Set here rather than inline so the escalation policy is one visible number,
# which is the honest version of what the 0.15 threshold was pretending to be.
CONFIDENCE_FLOOR = 0.6

# DriftGuard's default. Used only to compute what it *would* have done, which
# is how the suppression claim is measured.
DRIFTGUARD_THRESHOLD = 0.15


@dataclass
class Decision:
    """A verdict plus the analysis it was based on and what it displaces."""

    event_id: str
    model_id: str
    verdict: Verdict
    analysis: Analysis | None = None
    # True when DriftGuard's threshold would have fired but we did not act.
    driftguard_would_page: bool = False
    needs_human: bool = False
    raw: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "model_id": self.model_id,
            "verdict": self.verdict.to_dict(),
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "driftguard_would_page": self.driftguard_would_page,
            "needs_human": self.needs_human,
        }


def _heuristic_verdict(event: DriftEvent, analysis: Analysis) -> Verdict:
    """Deterministic action when Gemma output is unavailable or unparseable.

    Mirrors the policy stated in the evaluator system prompt so a degraded run
    still never retrains on data the heuristic flagged as pipeline-corrupted.
    """
    tax = analysis.taxonomy
    if tax in ("upstream_bug", "schema_break"):
        action, conf = "INVESTIGATE", 0.55
    elif tax == "seasonal":
        action, conf = "BENIGN", 0.5
    elif tax in ("covariate_shift", "concept_drift", "label_shift"):
        action, conf = "RETRAIN", 0.5
    else:
        action, conf = "INVESTIGATE", 0.3

    return Verdict(
        action=action,
        confidence=conf,
        reasoning=analysis.hypothesis,
        primary_features=analysis.primary_features,
        taxonomy=tax,
        source="heuristic",
    )


def decide(
    event: DriftEvent,
    client: GemmaClient | None = None,
    analysis: Analysis | None = None,
    *,
    run_analysis: bool = True,
) -> Decision:
    """UC 5 + 10 — decide what happens after a drift alert.

    Runs the reasoner first by default: a decision made without a cause is the
    threshold problem again in a different shape.
    """
    client = client or get_client()

    if analysis is None and run_analysis:
        analysis = analyze(event, client)

    parsed, raw = client.generate_json(
        evaluator_prompt(
            event,
            hypothesis=analysis.hypothesis if analysis else "",
            taxonomy=analysis.taxonomy or "" if analysis else "",
        ),
        SYSTEM_EVALUATOR,
    )

    if not parsed or not str(parsed.get("reasoning") or "").strip():
        verdict = _heuristic_verdict(event, analysis or _heuristic(event))
    else:
        feats = parsed.get("primary_features") or []
        if isinstance(feats, str):
            feats = [feats]
        feats = [f for f in (str(x) for x in feats) if f in event.feature_scores]

        verdict = Verdict(
            action=str(parsed.get("action", "INVESTIGATE")),
            confidence=parsed.get("confidence", 0.0),
            reasoning=str(parsed["reasoning"]).strip(),
            primary_features=feats or (analysis.primary_features if analysis else []),
            taxonomy=parsed.get("taxonomy") or (analysis.taxonomy if analysis else None),
            source=f"gemma:{client.model}",
        )

        # A model asked never to retrain on pipeline faults can still do it.
        # The guard is cheap and the failure mode is expensive, so enforce it
        # in code rather than trusting the instruction.
        if verdict.action == "RETRAIN" and verdict.taxonomy in (
            "upstream_bug",
            "schema_break",
        ):
            verdict.action = "INVESTIGATE"
            verdict.reasoning = (
                "[policy override: RETRAIN blocked on a pipeline-fault taxonomy] "
                + verdict.reasoning
            )
            verdict.confidence = min(verdict.confidence, CONFIDENCE_FLOOR - 0.01)

    verdict.suppressed = not verdict.acts
    would_page = event.global_drift_score >= DRIFTGUARD_THRESHOLD

    return Decision(
        event_id=event.event_id,
        model_id=event.model_id,
        verdict=verdict,
        analysis=analysis,
        driftguard_would_page=would_page and verdict.suppressed,
        needs_human=verdict.confidence < CONFIDENCE_FLOOR or verdict.action == "INVESTIGATE",
        raw=raw,
    )


@dataclass
class ChallengerVerdict:
    """UC 6 — promote / hold / reject a retrained challenger."""

    recommendation: str
    confidence: float
    reasoning: str
    blocking_evidence: str = ""
    report: ChallengerReport | None = None
    source: str = "gemma"

    def __post_init__(self) -> None:
        r = str(self.recommendation).strip().upper()
        self.recommendation = r if r in ("PROMOTE", "HOLD", "REJECT") else "HOLD"
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "blocking_evidence": self.blocking_evidence,
            "report": self.report.to_dict() if self.report else None,
            "source": self.source,
        }


def judge_challenger(
    report: ChallengerReport,
    event: DriftEvent | None = None,
    client: GemmaClient | None = None,
) -> ChallengerVerdict:
    """UC 6 — did the challenger improve on the segment that actually drifted?

    The overall metric is the number that misleads. A challenger can gain
    globally while regressing on exactly the segment it was retrained for.
    """
    client = client or get_client()
    parsed, _ = client.generate_json(challenger_prompt(report, event), SYSTEM_CHALLENGER)

    if not parsed or not str(parsed.get("reasoning") or "").strip():
        seg = report.segment_delta
        if seg is None:
            rec, conf, why = (
                "HOLD",
                0.4,
                f"Overall {report.metric_name} moved {report.overall_delta:+.4f}, but "
                f"performance on {report.segment_name} was not measured. The segment "
                f"that drifted is the one the retrain was meant to fix.",
            )
        elif seg < 0:
            rec, conf, why = (
                "REJECT",
                0.6,
                f"Challenger gains {report.overall_delta:+.4f} overall but loses "
                f"{seg:+.4f} on {report.segment_name} — it did not fix the drift "
                f"it was retrained for.",
            )
        else:
            rec, conf, why = (
                "PROMOTE",
                0.55,
                f"Challenger improves {report.overall_delta:+.4f} overall and "
                f"{seg:+.4f} on {report.segment_name}.",
            )
        return ChallengerVerdict(
            recommendation=rec,
            confidence=conf,
            reasoning=why,
            blocking_evidence="Heuristic fallback: Gemma was unavailable.",
            report=report,
            source="heuristic",
        )

    return ChallengerVerdict(
        recommendation=str(parsed.get("recommendation", "HOLD")),
        confidence=parsed.get("confidence", 0.0),
        reasoning=str(parsed["reasoning"]).strip(),
        blocking_evidence=str(parsed.get("blocking_evidence") or "").strip(),
        report=report,
        source=f"gemma:{client.model}",
    )


def gate(
    event: DriftEvent,
    train_fn: Callable[[], Any] | None = None,
    client: GemmaClient | None = None,
    on_suppress: Callable[[Decision], None] | None = None,
) -> tuple[Decision, Any]:
    """The ``@dg.retrainer`` seam — reason before retraining.

    Wire into the SDK without modifying it::

        dg = DriftGuard(model_id="fraud-v2", drift_threshold=0.15, auto_retrain=True)

        @dg.retrainer
        def gated_retrain():
            decision, model = gate(build_drift_event(dg), train_new_model)
            return model          # None when suppressed — the alert-fatigue win

    Returns (decision, train_fn result or None).
    """
    decision = decide(event, client)

    if not decision.verdict.acts or decision.needs_human:
        if on_suppress:
            on_suppress(decision)
        return decision, None

    return decision, (train_fn() if train_fn else None)


def measure_suppression(decisions: list[Decision]) -> dict[str, Any]:
    """Alert-fatigue numbers, measured from a real run.

    Every number in the README metric row comes from here (AGENTS.md §2) —
    nothing on screen is hardcoded.
    """
    total = len(decisions)
    if not total:
        return {
            "alerts_fired": 0,
            "actionable": 0,
            "suppressed": 0,
            "noise_reduction_pct": 0.0,
            "escalated_to_human": 0,
            "by_action": {},
            "by_source": {},
        }

    actionable = sum(1 for d in decisions if d.verdict.acts and not d.needs_human)
    by_action: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for d in decisions:
        by_action[d.verdict.action] = by_action.get(d.verdict.action, 0) + 1
        by_source[d.verdict.source] = by_source.get(d.verdict.source, 0) + 1

    gemma_backed = sum(1 for d in decisions if d.verdict.source.startswith("gemma"))

    return {
        # Every event in the corpus crossed DriftGuard's threshold, so this is
        # the page count the threshold alone would have produced.
        "alerts_fired": sum(1 for d in decisions if d.driftguard_would_page or d.verdict.acts),
        "actionable": actionable,
        "suppressed": total - actionable,
        "noise_reduction_pct": round((total - actionable) / total * 100, 1),
        "escalated_to_human": sum(1 for d in decisions if d.needs_human),
        "by_action": dict(sorted(by_action.items())),
        "by_source": dict(sorted(by_source.items())),
        # AGENTS.md §2: a run served by the heuristic fallback measures the
        # fallback, not Gemma. Anything below 100% here is not a result about
        # the product and must not be quoted as one.
        "gemma_backed_decisions": gemma_backed,
        "reportable": gemma_backed == total,
    }


def measure_accuracy(decisions: list[Decision], events: list[DriftEvent]) -> dict[str, Any]:
    """Taxonomy accuracy against hand-labelled ground truth.

    Only scenario fixtures carry ``ground_truth``, so this is reported as
    "N of M labelled events" and never extrapolated (AGENTS.md §2).
    """
    truth = {e.event_id: e.ground_truth for e in events if e.ground_truth}
    # Only Gemma-backed decisions can be scored. The heuristic fallback derives
    # its taxonomy from the same signals the fixtures were built from, so
    # grading it against those labels scores the fixture generator against
    # itself and trivially returns ~100%. That is not a result.
    scored = [
        d
        for d in decisions
        if d.event_id in truth and d.verdict.source.startswith("gemma")
    ]
    skipped = sum(
        1
        for d in decisions
        if d.event_id in truth and not d.verdict.source.startswith("gemma")
    )

    if not scored:
        return {
            "labelled_events": 0,
            "taxonomy_correct": 0,
            "taxonomy_accuracy_pct": None,
            "skipped_heuristic": skipped,
            "note": (
                "No Gemma-backed decisions to score. Heuristic-fallback verdicts "
                "are excluded by design: they are not measurable results."
            ),
        }

    correct = sum(1 for d in scored if d.verdict.taxonomy == truth[d.event_id])
    confusion: dict[str, dict[str, int]] = {}
    for d in scored:
        exp = truth[d.event_id]
        got = d.verdict.taxonomy or "none"
        confusion.setdefault(exp, {})
        confusion[exp][got] = confusion[exp].get(got, 0) + 1

    return {
        "labelled_events": len(scored),
        "taxonomy_correct": correct,
        "taxonomy_accuracy_pct": round(correct / len(scored) * 100, 1),
        "skipped_heuristic": skipped,
        "confusion": confusion,
    }
