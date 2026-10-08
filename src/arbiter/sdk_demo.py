"""Live integration proof: driftguard-ai-sdk -> Arbiter -> Gemma 4.

This is the end-to-end check that the product thesis actually holds. It uses
the real SDK (unmodified, from pip), trains a real model, drifts the data for
real, and sends the detector's real output through Gemma.

Nothing here is synthetic except the data generator:

    1. Train a logistic-regression fraud model on a reference population.
    2. Feed 200 in-distribution batches through the SDK's ADWIN detector.
    3. Shift the population (new high-value merchant segment) for 200 more.
    4. Read ``ADWINDriftDetector.get_status()`` — positional feature scores.
    5. Map indices -> names (the step without which narration is worthless).
    6. Measure real champion accuracy before and after the shift.
    7. Send it all to Gemma and print the verdict.

Run:  python -m arbiter.sdk_demo
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .evaluator import decide
from .gemma_client import get_client
from .reasoner import meets_quality_bar
from .schema import DriftEvent, map_feature_scores

FEATURES = [
    "transaction_amount",
    "merchant_category",
    "hour_of_day",
    "card_present",
    "account_age_days",
    "velocity_1h",
]

REF_MEAN = np.array([240.0, 8.4, 14.2, 0.62, 1240.0, 1.8])
REF_STD = np.array([180.0, 3.1, 5.8, 0.48, 820.0, 1.4])


def _label(x: np.ndarray) -> np.ndarray:
    """Ground-truth fraud rule: large amounts, card-not-present, high velocity.

    A fixed rule, so a population shift genuinely moves the decision boundary
    rather than only moving the inputs.
    """
    score = (
        0.004 * x[:, 0]  # transaction_amount
        + 1.1 * (1 - x[:, 3])  # card not present
        + 0.45 * x[:, 5]  # velocity_1h
        - 0.0008 * x[:, 4]  # account_age_days (older = safer)
    )
    return (score > 2.2).astype(int)


def _stats(sample: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-feature mean/std/null_rate, keyed by name."""
    return {
        name: {
            "mean": round(float(sample[:, i].mean()), 4),
            "std": round(float(sample[:, i].std()), 4),
            "null_rate": round(float(np.isnan(sample[:, i]).mean()), 4),
        }
        for i, name in enumerate(FEATURES)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live SDK -> Arbiter -> Gemma proof.")
    parser.add_argument("--batches", type=int, default=200, help="batches per phase")
    parser.add_argument("--json", metavar="PATH", help="write the result to PATH")
    args = parser.parse_args(argv)

    try:
        from driftguard.drift_detector import ADWINDriftDetector
    except ImportError:
        print("driftguard-ai-sdk is not installed. pip install -r requirements.txt")
        return 1
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn is required for this demo. pip install scikit-learn")
        return 1

    rng = np.random.default_rng(42)

    print("=" * 70)
    print("STEP 1  Train a champion model on the reference population")
    print("=" * 70)
    ref = rng.normal(REF_MEAN, REF_STD, size=(4000, 6))
    y_ref = _label(ref)
    scaler = StandardScaler().fit(ref)
    champion = LogisticRegression(max_iter=1000).fit(scaler.transform(ref), y_ref)

    holdout = rng.normal(REF_MEAN, REF_STD, size=(1500, 6))
    acc_ref = float(champion.score(scaler.transform(holdout), _label(holdout)))
    print(f"  training rows      : {len(ref)}")
    print(f"  fraud base rate    : {y_ref.mean():.1%}")
    print(f"  accuracy (in-dist) : {acc_ref:.4f}")

    print()
    print("=" * 70)
    print("STEP 2  Feed in-distribution traffic through the SDK's ADWIN detector")
    print("=" * 70)
    detector = ADWINDriftDetector(num_features=len(FEATURES), reference_data=ref)
    for _ in range(args.batches):
        detector.update(rng.normal(REF_MEAN, REF_STD))
    quiet = detector.get_status()
    print(f"  global_drift_score : {quiet['global_drift_score']:.4f}")
    print(f"  drift_detected     : {quiet['drift_detected']}")

    print()
    print("=" * 70)
    print("STEP 3  Shift the population: new high-value merchant segment")
    print("=" * 70)
    drift_mean = REF_MEAN.copy()
    drift_std = REF_STD.copy()
    drift_mean[0] = 890.0  # transaction_amount 240 -> 890
    drift_std[0] = 310.0
    drift_mean[1] = 13.9  # merchant_category 8.4 -> 13.9
    for _ in range(args.batches):
        detector.update(rng.normal(drift_mean, drift_std))

    status = detector.get_status()
    print(f"  global_drift_score : {status['global_drift_score']:.4f}")
    print(f"  drift_detected     : {status['drift_detected']}")
    print(f"  feature_scores type: {type(status['feature_scores']).__name__} "
          f"(positional, no names)")
    print(f"  raw                : {[round(float(s), 4) for s in status['feature_scores']]}")

    print()
    print("=" * 70)
    print("STEP 4  Map positional scores -> feature names (the critical step)")
    print("=" * 70)
    named = map_feature_scores(status["feature_scores"], FEATURES)
    for name, score in sorted(named.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(score * 40)
        print(f"  {name:20} {score:.4f} {bar}")

    print()
    print("=" * 70)
    print("STEP 5  Measure the real cost of the drift on the champion model")
    print("=" * 70)
    drifted = rng.normal(drift_mean, drift_std, size=(1500, 6))
    acc_drift = float(champion.score(scaler.transform(drifted), _label(drifted)))
    print(f"  accuracy on reference population : {acc_ref:.4f}")
    print(f"  accuracy on drifted population   : {acc_drift:.4f}")
    print(f"  delta                            : {acc_drift - acc_ref:+.4f}")

    event = DriftEvent(
        model_id="fraud-detector-v2",
        global_drift_score=round(float(status["global_drift_score"]), 4),
        feature_scores={k: round(v, 4) for k, v in named.items()},
        drift_detected=bool(status["drift_detected"]),
        reference_stats=_stats(ref),
        current_stats=_stats(drifted),
        event_id="sdk-live-001",
        context={
            "source": "driftguard-ai-sdk 1.0.4 ADWINDriftDetector (live)",
            "upstream_jobs": "all green",
            "schema_version": "unchanged",
            "deploy_markers": "no model deploy in 14 days",
            "champion_accuracy_reference": round(acc_ref, 4),
            "champion_accuracy_current": round(acc_drift, 4),
        },
    )

    print()
    print("=" * 70)
    print("STEP 6  Send real SDK telemetry to Gemma 4")
    print("=" * 70)
    client = get_client()
    print(f"  model     : {client.model}")
    print(f"  available : {client.available}")
    if not client.available:
        print("\n  Gemma is unavailable. Set GEMMA_API_KEY in .env and re-run;")
        print("  a heuristic verdict here would not demonstrate the product.")
        return 1

    decision = decide(event, client)
    v = decision.verdict
    print()
    print(f"  ACTION      : {v.action}")
    print(f"  CONFIDENCE  : {v.confidence:.2f}")
    print(f"  TAXONOMY    : {v.taxonomy}")
    print(f"  FEATURES    : {', '.join(v.primary_features) or '(none)'}")
    print(f"  NEEDS HUMAN : {decision.needs_human}")
    print(f"  SOURCE      : {v.source}")
    print()
    print("  REASONING:")
    for line in _wrap(v.reasoning, 66):
        print(f"    {line}")

    if decision.analysis:
        print()
        print(f"  ROOT CAUSE (taxonomy: {decision.analysis.taxonomy}):")
        for line in _wrap(decision.analysis.hypothesis, 66):
            print(f"    {line}")
        print()
        print("  QUALITY BAR:")
        for criterion, passed in meets_quality_bar(decision.analysis, event).items():
            print(f"    {'PASS' if passed else 'FAIL'}  {criterion}")

    print()
    print("=" * 70)
    gemma_backed = v.source.startswith("gemma")
    print("RESULT: live SDK telemetry reasoned over by Gemma 4"
          if gemma_backed else "RESULT: heuristic fallback — NOT a product result")
    print("=" * 70)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "sdk_version": "1.0.4",
                    "sdk_raw_status": {
                        "global_drift_score": float(status["global_drift_score"]),
                        "feature_scores": [float(s) for s in status["feature_scores"]],
                        "drift_detected": bool(status["drift_detected"]),
                    },
                    "named_feature_scores": named,
                    "champion_accuracy_reference": acc_ref,
                    "champion_accuracy_current": acc_drift,
                    "event": event.to_dict(),
                    "decision": decision.to_dict(),
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.json}")

    return 0 if gemma_backed else 1


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    sys.exit(main())
