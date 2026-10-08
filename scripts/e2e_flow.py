"""End-to-end flow check: account -> API key -> 3 models -> drift telemetry.

Exercises the exact path a user takes, over HTTP, so everything the dashboard
calls is covered. Run it against a scratch database:

    ARBITER_DB=./e2e.db python scripts/e2e_flow.py

Three models, three different drift causes, because a reasoning layer that
cannot tell them apart is not worth having:

    fraud-detector-v2   genuine population shift   -> retraining helps
    churn-predictor-v1  upstream pipeline fault     -> retraining would corrupt
    credit-risk-v3      concept drift               -> inputs stable, perf falls
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from fastapi.testclient import TestClient

from arbiter import auth, keys
from arbiter.api import app

# Credentials come from the environment. A password in version control is a
# committed credential even when the account is only a local test one
# (AGENTS.md §7).
EMAIL = os.getenv("ARBITER_TEST_EMAIL", "e2e@example.com")
PASSWORD = os.getenv("ARBITER_TEST_PASSWORD", "e2e-local-password")

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

REF_BATCH = 150   # a window an analyst would call conclusive
DRIFT_BATCH = 120
REASONING_TIMEOUT_S = 300


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


def main() -> int:
    keys.reset_db()
    auth.ensure_auth_schema()
    client = TestClient(app)
    failures: list[str] = []

    banner("STEP 1  Register an account")
    res = client.post(
        "/auth/register", json={"email": EMAIL, "password": PASSWORD, "name": "Yugen"}
    )
    print(f"  POST /auth/register -> {res.status_code}")
    if res.status_code != 200:
        print("  FAILED:", res.text)
        return 1
    print(f"  user id {res.json()['user']['id']} · {res.json()['user']['email']}")

    banner("STEP 2  Sign in")
    res = client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
    print(f"  POST /auth/login -> {res.status_code}")
    if res.status_code != 200:
        print("  FAILED:", res.text)
        return 1
    AUTH = {"Authorization": f"Bearer {res.json()['token']}"}
    print(f"  GET  /auth/me    -> {client.get('/auth/me', headers=AUTH).json()['user']['email']}")

    bad = client.post("/auth/login", json={"email": EMAIL, "password": "wrong"})
    print(f"  wrong password   -> {bad.status_code} (expect 401)")
    if bad.status_code != 401:
        failures.append("wrong password was not rejected")

    banner("STEP 3  Generate an API key")
    res = client.post("/api/keys", headers=AUTH, json={"label": "local-dev"})
    if res.status_code != 200:
        print("  FAILED:", res.text)
        return 1
    api_key = res.json()["key"]
    print(f"  key hint  : {res.json()['key_hint']}")
    print(f"  setup url : {res.json()['setup']['DRIFTGUARD_API_URL']}")
    listed = client.get("/api/keys", headers=AUTH).json()["keys"]
    leaked = any(api_key in str(k) for k in listed)
    print(f"  plaintext recoverable from the list? {leaked} (must be False)")
    if leaked:
        failures.append("API key plaintext is recoverable")
    SDK = {"X-API-Key": api_key}

    banner("STEP 4  Register 3 models with that key")
    for mid, spec in MODELS.items():
        res = client.post(
            "/models/register",
            headers=SDK,
            json={
                "model_id": mid,
                "features": spec["features"],
                "drift_threshold": 0.15,
                "version": "1.0.0",
            },
        )
        body = res.json()
        print(
            f"  {mid:20} -> {res.status_code}  "
            f"{body.get('features_registered')} features, reasoning={body.get('reasoning_enabled')}"
        )
        if res.status_code != 200:
            failures.append(f"{mid} registration failed")

    owned = client.get("/api/models", headers=AUTH).json()
    print(f"  GET /api/models -> {owned['count']} models owned by this account")
    if owned["count"] != 3:
        failures.append(f"expected 3 models, got {owned['count']}")

    banner("STEP 5  Stream telemetry: reference window, then drift")
    rng = np.random.default_rng(42)

    for mid, spec in MODELS.items():
        ref_m = np.array(spec["ref_mean"])
        ref_s = np.array(spec["ref_std"])

        for _ in range(REF_BATCH):
            client.post(
                f"/predict/{mid}",
                headers=SDK,
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
            # A pipeline fault collapses variance toward a default; a real
            # population shift widens it. That contrast is the signal.
            drift_s[idx] = ref_s[idx] * (0.03 if spec["kind"] == "pipeline_break" else 1.6)

        queued = 0
        for i in range(DRIFT_BATCH):
            vec = rng.normal(drift_m, drift_s).tolist()
            if spec["kind"] == "pipeline_break" and i % 3 == 0:
                vec[next(iter(spec["shift"]))] = None   # nulls: the tell
            # Concept drift shows little feature movement but still crosses
            # the threshold — the case a distribution detector reads wrong.
            score = 0.28 if spec["kind"] == "concept_drift" else float(rng.uniform(0.6, 0.9))
            r = client.post(
                f"/predict/{mid}",
                headers=SDK,
                json={
                    "model_id": mid,
                    "features": vec,
                    "prediction": [1],
                    "drift_score": score,
                },
            )
            if r.json().get("reasoning_queued"):
                queued += 1

        total = client.get(f"/api/models/{mid}/telemetry", headers=AUTH).json()["total"]
        print(f"  {mid:20} telemetry={total:3}  reasoning_queued={queued:2}  ({spec['kind']})")
        if total < REF_BATCH + DRIFT_BATCH:
            failures.append(f"{mid} lost telemetry ({total} rows)")

    banner("STEP 6  Wait for background reasoning (two Gemma calls per event)")
    deadline = time.time() + REASONING_TIMEOUT_S
    reasoned: set[str] = set()
    while time.time() < deadline:
        ev = client.get("/api/live/events", headers=AUTH).json()
        reasoned = {e["model_id"] for e in ev["events"]}
        print(f"  events={ev['count']:2}  reasoned: {sorted(reasoned)}", flush=True)
        if len(reasoned) >= len(MODELS):
            break
        time.sleep(10)

    missing = set(MODELS) - reasoned
    if missing:
        failures.append(f"no drift event reasoned for: {sorted(missing)}")

    banner("STEP 7  Per-model metrics and verdicts")
    for mid in MODELS:
        m = client.get(f"/api/models/{mid}/metrics", headers=AUTH).json()
        evs = client.get(f"/api/models/{mid}/events", headers=AUTH).json()
        print(f"\n  {mid}")
        print(f"    telemetry rows : {m['telemetry_count']}")
        print(f"    features named : {len(m['features'])}")
        print(f"    latest drift   : {m['latest_drift_score']}")
        print(f"    drift events   : {evs['count']}")
        print(f"    by action      : {m['by_action']}")
        print(f"    by taxonomy    : {m['by_taxonomy']}")
        print(f"    reportable     : {m['reportable']}")
        for e in evs["events"][:1]:
            print(f"    verdict        : {e['action']} @ {e['confidence']} ({e['taxonomy']})")
            print(f"    reasoning      : {(e['reasoning'] or '')[:160]}")
        if evs["count"] == 0:
            failures.append(f"{mid} produced no verdict")

    banner("STEP 8  Per-model chat and fleet chat")
    r = client.post(
        f"/api/models/{'fraud-detector-v2'}/chat",
        headers=AUTH,
        json={"question": "Why did this model drift, and should I retrain?"},
    ).json()
    print(f"  model chat [{r.get('source')}] context_events={r.get('context_events')}")
    print(f"    {(r.get('answer') or '')[:300]}")
    if not str(r.get("source", "")).startswith("gemma"):
        failures.append("model chat did not reach Gemma")

    r = client.post(
        "/api/chat",
        headers=AUTH,
        json={"question": "Which of my models is in the worst shape and why?"},
    ).json()
    print(f"  fleet chat [{r.get('source')}] context_events={r.get('context_events')}")
    print(f"    {(r.get('answer') or '')[:300]}")
    if not str(r.get("source", "")).startswith("gemma"):
        failures.append("fleet chat did not reach Gemma")

    banner("STEP 9  Account isolation")
    client.post("/auth/register", json={"email": "other@example.com", "password": "otherpass123"})
    other = client.post(
        "/auth/login", json={"email": "other@example.com", "password": "otherpass123"}
    ).json()["token"]
    OTHER = {"Authorization": f"Bearer {other}"}
    om = client.get("/api/models", headers=OTHER).json()["count"]
    oe = client.get("/api/live/events", headers=OTHER).json()["count"]
    print(f"  second account sees {om} models, {oe} events (both must be 0)")
    if om or oe:
        failures.append("telemetry leaked across accounts")

    banner("STEP 10  Settings: change email with password")
    r = client.patch(
        "/auth/email",
        headers=AUTH,
        json={"new_email": "yugen.updated@example.com", "password": PASSWORD},
    )
    print(f"  PATCH /auth/email           -> {r.status_code} {r.json().get('user', {}).get('email', '')}")
    bad = client.patch(
        "/auth/email", headers=AUTH, json={"new_email": "x@y.com", "password": "wrong"}
    )
    print(f"  with the wrong password     -> {bad.status_code} (expect 400)")
    if r.status_code != 200:
        failures.append("email change failed")
    if bad.status_code != 400:
        failures.append("email change accepted a wrong password")
    client.patch("/auth/email", headers=AUTH, json={"new_email": EMAIL, "password": PASSWORD})

    if failures:
        banner("RESULT: FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1

    banner("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
