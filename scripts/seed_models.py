"""Register three models against a RUNNING Arbiter and stream drift telemetry.

Unlike scripts/e2e_flow.py (which drives the app in-process via TestClient),
this talks to a live server over HTTP — the same path a user's own code takes.

    ARBITER_TOKEN=<session>  ARBITER_KEY=<ak_live_...>  python scripts/seed_models.py

Three models, three causes, so the dashboard has something real to show:

    fraud-detector-v2   genuine population shift
    churn-predictor-v1  upstream pipeline fault (nulls + variance collapse)
    credit-risk-v3      concept drift (inputs stable, performance falls)
"""

from __future__ import annotations

import os
import sys
import time

import httpx
import numpy as np

BASE = os.getenv("ARBITER_URL", "http://127.0.0.1:8000")
TOKEN = os.getenv("ARBITER_TOKEN", "")
KEY = os.getenv("ARBITER_KEY", "")

MODELS: dict[str, dict] = {
    "fraud-detector-v2": {
        "features": [
            "transaction_amount",
            "merchant_category",
            "hour_of_day",
            "card_present",
            "account_age_days",
            "velocity_1h",
        ],
        "ref_mean": [240.0, 8.4, 14.2, 0.62, 1240.0, 1.8],
        "ref_std": [180.0, 3.1, 5.8, 0.48, 820.0, 1.4],
        "kind": "new_segment",
        "shift": {0: 3.7, 1: 1.65},
    },
    "churn-predictor-v1": {
        "features": [
            "monthly_charges",
            "tenure_months",
            "support_tickets",
            "data_usage_gb",
            "contract_type",
            "late_payments",
        ],
        "ref_mean": [64.8, 32.4, 1.2, 18.6, 1.7, 0.4],
        "ref_std": [30.2, 24.1, 1.6, 12.4, 0.8, 0.9],
        "kind": "pipeline_break",
        "shift": {4: 0.04},
    },
    "credit-risk-v3": {
        "features": [
            "income_annual",
            "debt_to_income",
            "credit_utilization",
            "inquiries_6mo",
            "employment_years",
            "open_accounts",
        ],
        "ref_mean": [58400.0, 0.34, 0.41, 1.6, 6.2, 5.4],
        "ref_std": [26200.0, 0.14, 0.22, 1.5, 4.8, 2.9],
        "kind": "concept_drift",
        "shift": {},
    },
}

REF_BATCH = 90
DRIFT_BATCH = 60


def main() -> int:
    if not KEY:
        print("Set ARBITER_KEY (mint one in the dashboard's Settings tab).")
        return 1

    sdk = {"X-API-Key": KEY}
    auth = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    rng = np.random.default_rng(42)

    with httpx.Client(base_url=BASE, timeout=30.0) as c:
        print("Registering models")
        for mid, spec in MODELS.items():
            r = c.post(
                "/models/register",
                headers=sdk,
                json={
                    "model_id": mid,
                    "features": spec["features"],
                    "drift_threshold": 0.15,
                    "version": "1.0.0",
                },
            )
            print(f"  {mid:20} {r.status_code} {r.json().get('message', '')}")
            if r.status_code != 200:
                return 1

        print("\nStreaming telemetry")
        for mid, spec in MODELS.items():
            ref_m = np.array(spec["ref_mean"])
            ref_s = np.array(spec["ref_std"])

            for _ in range(REF_BATCH):
                c.post(
                    f"/predict/{mid}",
                    headers=sdk,
                    json={
                        "model_id": mid,
                        "features": rng.normal(ref_m, ref_s).tolist(),
                        "prediction": [0],
                        "drift_score": float(rng.uniform(0.02, 0.09)),
                    },
                )

            drift_m, drift_s = ref_m.copy(), ref_s.copy()
            for idx, mult in spec["shift"].items():
                drift_m[idx] = ref_m[idx] * mult
                # A pipeline fault collapses variance toward a default value;
                # a real population shift widens it. That contrast is the
                # signal the reasoner is meant to read.
                drift_s[idx] = ref_s[idx] * (
                    0.03 if spec["kind"] == "pipeline_break" else 1.6
                )

            queued = 0
            for i in range(DRIFT_BATCH):
                vec = rng.normal(drift_m, drift_s).tolist()
                if spec["kind"] == "pipeline_break" and i % 3 == 0:
                    vec[next(iter(spec["shift"]))] = None   # nulls: the tell
                score = (
                    0.28
                    if spec["kind"] == "concept_drift"
                    else float(rng.uniform(0.6, 0.9))
                )
                r = c.post(
                    f"/predict/{mid}",
                    headers=sdk,
                    json={
                        "model_id": mid,
                        "features": vec,
                        "prediction": [1],
                        "drift_score": score,
                    },
                )
                if r.json().get("reasoning_queued"):
                    queued += 1

            print(f"  {mid:20} {REF_BATCH + DRIFT_BATCH} rows, reasoning_queued={queued}")

        if not auth:
            print("\nNo ARBITER_TOKEN given; skipping verification.")
            return 0

        print("\nWaiting for background reasoning")
        deadline = time.time() + 300
        while time.time() < deadline:
            ev = c.get("/api/live/events", headers=auth).json()
            done = sorted({e["model_id"] for e in ev["events"]})
            print(f"  events={ev['count']:2}  reasoned: {done}", flush=True)
            if len(done) >= len(MODELS):
                break
            time.sleep(12)

        print("\nVerdicts")
        for mid in MODELS:
            m = c.get(f"/api/models/{mid}/metrics", headers=auth).json()
            evs = c.get(f"/api/models/{mid}/events", headers=auth).json()
            print(f"\n  {mid}")
            print(f"    telemetry   : {m['telemetry_count']}")
            print(f"    events      : {evs['count']}")
            print(f"    by action   : {m['by_action']}")
            print(f"    by taxonomy : {m['by_taxonomy']}")
            for e in evs["events"][:1]:
                print(f"    verdict     : {e['action']} @ {e['confidence']} ({e['taxonomy']})")
                print(f"    features    : {', '.join(e['primary_features']) or '—'}")
                print(f"    reasoning   : {(e['reasoning'] or '')[:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
