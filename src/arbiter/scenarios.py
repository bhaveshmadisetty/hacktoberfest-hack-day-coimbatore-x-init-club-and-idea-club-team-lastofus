"""Synthetic drift scenarios with hand-labelled causes.

These are the replay corpus. Four scenario families, each generating drift that
a threshold cannot tell apart but a reasoner should:

- ``new_segment``   — real population shift. Retraining helps.
- ``pipeline_break``— an upstream join started emitting nulls/zeros. Retraining
                      on this data bakes the bug into the model.
- ``seasonal``      — month-end batch pattern seen in prior cycles. No action.
- ``concept_drift`` — inputs stable, the input→label relationship moved. The
                      case a feature-distribution detector is worst at.

``drift_detected`` is True for every event: that is the point. DriftGuard fires
on all of them, so the suppression count measured on this corpus is the real
alert-fatigue reduction, not a number chosen to look good.

Each event carries ``ground_truth`` — never shown to the model — so accuracy
claims trace to an actual comparison (AGENTS.md §2).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .schema import DriftEvent

# Feature sets per model, in the positional order the detector would see them.
# Keeping real column names here is what lets a narration say
# "transaction_amount", which is the whole quality bar.
FRAUD_FEATURES = [
    "transaction_amount",
    "merchant_category",
    "hour_of_day",
    "card_present",
    "account_age_days",
    "velocity_1h",
]

CHURN_FEATURES = [
    "monthly_charges",
    "tenure_months",
    "support_tickets",
    "data_usage_gb",
    "contract_type",
    "late_payments",
]

CREDIT_FEATURES = [
    "income_annual",
    "debt_to_income",
    "credit_utilization",
    "inquiries_6mo",
    "employment_years",
    "open_accounts",
]

MODELS: dict[str, list[str]] = {
    "fraud-detector-v2": FRAUD_FEATURES,
    "churn-predictor-v1": CHURN_FEATURES,
    "credit-risk-v3": CREDIT_FEATURES,
}

# Plausible reference distributions per feature.
_BASELINES: dict[str, tuple[float, float]] = {
    "transaction_amount": (240.0, 180.0),
    "merchant_category": (8.4, 3.1),
    "hour_of_day": (14.2, 5.8),
    "card_present": (0.62, 0.48),
    "account_age_days": (1240.0, 820.0),
    "velocity_1h": (1.8, 1.4),
    "monthly_charges": (64.8, 30.2),
    "tenure_months": (32.4, 24.1),
    "support_tickets": (1.2, 1.6),
    "data_usage_gb": (18.6, 12.4),
    "contract_type": (1.7, 0.8),
    "late_payments": (0.4, 0.9),
    "income_annual": (58400.0, 26200.0),
    "debt_to_income": (0.34, 0.14),
    "credit_utilization": (0.41, 0.22),
    "inquiries_6mo": (1.6, 1.5),
    "employment_years": (6.2, 4.8),
    "open_accounts": (5.4, 2.9),
}


def _ref_stats(features: list[str]) -> dict[str, dict[str, float]]:
    return {
        f: {"mean": _BASELINES[f][0], "std": _BASELINES[f][1], "null_rate": 0.0}
        for f in features
    }


def _quiet_scores(features: list[str], rng: random.Random) -> dict[str, float]:
    """Low baseline drift on every feature, so the signal has to stand out."""
    return {f: round(rng.uniform(0.02, 0.14), 3) for f in features}


def _jitter(
    current: dict[str, dict[str, float]], features: list[str], rng: random.Random
) -> None:
    """Small natural movement on non-drifting features, in place."""
    for f in features:
        s = current[f]
        s["mean"] = round(s["mean"] * rng.uniform(0.97, 1.03), 3)
        s["std"] = round(s["std"] * rng.uniform(0.96, 1.04), 3)


# ---- scenario builders ----------------------------------------------------


def new_segment(model_id: str, rng: random.Random) -> DriftEvent:
    """A new customer/merchant population enters the data. Retraining helps."""
    features = MODELS[model_id]
    ref = _ref_stats(features)
    cur = {f: dict(v) for f, v in ref.items()}
    scores = _quiet_scores(features, rng)

    # Two correlated features move together — one root cause, not two.
    primary, secondary = features[0], features[1]
    mult = rng.uniform(2.8, 4.2)
    cur[primary]["mean"] = round(ref[primary]["mean"] * mult, 2)
    cur[primary]["std"] = round(ref[primary]["std"] * rng.uniform(1.3, 1.9), 2)
    cur[secondary]["mean"] = round(ref[secondary]["mean"] * rng.uniform(1.4, 1.9), 2)
    scores[primary] = round(rng.uniform(0.88, 0.96), 3)
    scores[secondary] = round(rng.uniform(0.62, 0.78), 3)
    _jitter(cur, features[2:], rng)

    return DriftEvent(
        model_id=model_id,
        global_drift_score=round(rng.uniform(0.78, 0.9), 3),
        feature_scores=scores,
        drift_detected=True,
        reference_stats=ref,
        current_stats=cur,
        context={
            "upstream_jobs": "all green",
            "schema_version": "unchanged",
            "deploy_markers": "no model deploy in 14 days",
            "note": "marketing launched a new acquisition channel 3 days ago",
        },
        ground_truth="covariate_shift",
    )


def pipeline_break(model_id: str, rng: random.Random) -> DriftEvent:
    """An upstream join broke. Retraining on this corrupts the model."""
    features = MODELS[model_id]
    ref = _ref_stats(features)
    cur = {f: dict(v) for f, v in ref.items()}
    scores = _quiet_scores(features, rng)

    # The tell: nulls appear and variance collapses toward a default value.
    broken = features[4]
    cur[broken]["mean"] = round(ref[broken]["mean"] * rng.uniform(0.02, 0.08), 3)
    cur[broken]["std"] = round(ref[broken]["std"] * rng.uniform(0.01, 0.05), 3)
    cur[broken]["null_rate"] = round(rng.uniform(0.34, 0.61), 3)
    scores[broken] = round(rng.uniform(0.9, 0.98), 3)

    collateral = features[5]
    cur[collateral]["mean"] = round(ref[collateral]["mean"] * rng.uniform(0.4, 0.7), 3)
    cur[collateral]["null_rate"] = round(rng.uniform(0.1, 0.25), 3)
    scores[collateral] = round(rng.uniform(0.55, 0.72), 3)
    _jitter(cur, features[:4], rng)

    return DriftEvent(
        model_id=model_id,
        global_drift_score=round(rng.uniform(0.82, 0.94), 3),
        feature_scores=scores,
        drift_detected=True,
        reference_stats=ref,
        current_stats=cur,
        context={
            "upstream_jobs": f"warehouse_enrichment_dag FAILED 2 runs ago",
            "schema_version": "unchanged",
            "deploy_markers": "no model deploy in 9 days",
            "note": "on-call acknowledged an ETL alert this morning",
        },
        ground_truth="upstream_bug",
    )


def seasonal(model_id: str, rng: random.Random) -> DriftEvent:
    """Month-end batch pattern, matching prior cycles. No action warranted."""
    features = MODELS[model_id]
    ref = _ref_stats(features)
    cur = {f: dict(v) for f, v in ref.items()}
    scores = _quiet_scores(features, rng)

    cyclical = features[2]
    cur[cyclical]["mean"] = round(ref[cyclical]["mean"] * rng.uniform(1.25, 1.5), 3)
    cur[cyclical]["std"] = round(ref[cyclical]["std"] * rng.uniform(1.05, 1.2), 3)
    scores[cyclical] = round(rng.uniform(0.34, 0.48), 3)
    _jitter(cur, [f for f in features if f != cyclical], rng)

    return DriftEvent(
        model_id=model_id,
        global_drift_score=round(rng.uniform(0.3, 0.42), 3),
        feature_scores=scores,
        drift_detected=True,  # above DriftGuard's 0.15 threshold — it pages
        reference_stats=ref,
        current_stats=cur,
        context={
            "upstream_jobs": "all green",
            "schema_version": "unchanged",
            "deploy_markers": "no model deploy in 21 days",
            "note": (
                "month-end billing batch; same shift recorded in the "
                "previous 3 month-end cycles, each time self-resolving in 48h"
            ),
        },
        ground_truth="seasonal",
    )


def concept_drift(model_id: str, rng: random.Random) -> DriftEvent:
    """Inputs stable, input-to-label relationship moved. Retraining helps."""
    features = MODELS[model_id]
    ref = _ref_stats(features)
    cur = {f: dict(v) for f, v in ref.items()}
    # Feature drift is genuinely low — the damage shows in performance, not
    # in the input distributions. This is the case a detector reads wrong.
    scores = {f: round(rng.uniform(0.08, 0.22), 3) for f in features}
    _jitter(cur, features, rng)

    return DriftEvent(
        model_id=model_id,
        global_drift_score=round(rng.uniform(0.22, 0.31), 3),
        feature_scores=scores,
        drift_detected=True,
        reference_stats=ref,
        current_stats=cur,
        context={
            "upstream_jobs": "all green",
            "schema_version": "unchanged",
            "deploy_markers": "no model deploy in 30 days",
            "note": (
                "input distributions near-stable but live precision fell "
                "0.91 -> 0.74 over 10 days; fraud ring adapted to the "
                "existing decision boundary"
            ),
            "live_precision_ref": 0.91,
            "live_precision_current": 0.74,
        },
        ground_truth="concept_drift",
    )


SCENARIOS: dict[str, Callable[[str, random.Random], DriftEvent]] = {
    "new_segment": new_segment,
    "pipeline_break": pipeline_break,
    "seasonal": seasonal,
    "concept_drift": concept_drift,
}

# Weighted toward benign/noisy causes, because that is what an on-call rotation
# actually sees — and it is what makes the suppression number meaningful.
# Interleaved rather than grouped so any prefix of the corpus still covers all
# four causes: a short `--events 6` run must exercise every branch, not six
# seasonal events in a row.
_MIX = [
    "seasonal",
    "pipeline_break",
    "new_segment",
    "seasonal",
    "concept_drift",
    "pipeline_break",
    "seasonal",
    "new_segment",
    "seasonal",
    "pipeline_break",
    "concept_drift",
    "new_segment",
    "seasonal",
    "pipeline_break",
    "new_segment",
    "seasonal",
    "concept_drift",
    "pipeline_break",
    "new_segment",
    "seasonal",
]


def generate_replay(n: int = 20, seed: int = 20261008) -> list[DriftEvent]:
    """Build a deterministic replay corpus of n drift events.

    Seeded so the suppression numbers in the README are reproducible: anyone
    can re-run this and get the same corpus.
    """
    rng = random.Random(seed)
    model_ids = list(MODELS)
    events: list[DriftEvent] = []

    # Timestamps run backwards from now so the newest event sorts first and
    # "this week" queries (UC 11) have a real window to filter on.
    base = datetime.now(timezone.utc).replace(microsecond=0)

    for i in range(n):
        kind = _MIX[i % len(_MIX)]
        model_id = model_ids[i % len(model_ids)]
        event = SCENARIOS[kind](model_id, rng)
        event.event_id = f"evt-{i + 1:03d}"
        event.timestamp = (base - timedelta(hours=6 * i)).isoformat()
        event.context["scenario"] = kind
        events.append(event)

    return events


def summarize(events: list[DriftEvent]) -> dict[str, Any]:
    """Corpus composition — cited in the README so the mix is auditable."""
    counts: dict[str, int] = {}
    for e in events:
        counts[e.ground_truth or "unknown"] = counts.get(e.ground_truth or "unknown", 0) + 1
    return {
        "total_events": len(events),
        "all_fired_by_detector": all(e.drift_detected for e in events),
        "by_ground_truth": dict(sorted(counts.items())),
        "models": sorted({e.model_id for e in events}),
    }
