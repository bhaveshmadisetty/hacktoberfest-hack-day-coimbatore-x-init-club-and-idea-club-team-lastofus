"""FastAPI layer — what the dashboard talks to.

One in-process run holds the corpus and its decisions. Reasoning is computed
once, lazily, on the first request that needs it, then reused: a demo must not
re-call a hosted model on every page load, and the metric row has to agree with
the cards it sits above.

Endpoints:
    GET  /health              liveness + Gemma availability
    GET  /api/metrics         measured suppression numbers (never hardcoded)
    GET  /api/events          triage feed
    GET  /api/events/{id}     event detail with analysis and challenger judgment
    POST /api/ask             UC 11 natural-language query
    POST /api/reason          reason over a caller-supplied DriftEvent
    POST /api/replay/reset    recompute the run (demo pre-warm)
"""

from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__, keys
from .evaluator import (
    CONFIDENCE_FLOOR,
    DRIFTGUARD_THRESHOLD,
    Decision,
    decide,
    judge_challenger,
    measure_accuracy,
    measure_suppression,
)
from .gemma_client import get_client
from .ingest import router as sdk_router
from .query import EXAMPLE_QUESTIONS, ask, build_digest
from .reasoner import analyze, meets_quality_bar
from .scenarios import generate_replay, summarize
from .schema import ChallengerReport, DriftEvent

REPLAY_SIZE = int(os.getenv("ARBITER_REPLAY_SIZE", "20"))


class RunStore:
    """The current replay run: events, decisions, and derived numbers.

    Lock-guarded because uvicorn serves requests concurrently and the first
    two requests would otherwise both pay for a full reasoning pass.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[DriftEvent] = generate_replay(REPLAY_SIZE)
        self.decisions: list[Decision] = []
        self._reasoned = False

    def ensure_reasoned(self) -> list[Decision]:
        with self._lock:
            if not self._reasoned:
                client = get_client()
                self.decisions = [decide(e, client) for e in self.events]
                self._reasoned = True
            return self.decisions

    def reset(self, size: int | None = None) -> None:
        with self._lock:
            self.events = generate_replay(size or REPLAY_SIZE)
            self.decisions = []
            self._reasoned = False

    def find(self, event_id: str) -> tuple[DriftEvent, Decision] | None:
        decisions = self.ensure_reasoned()
        for event, decision in zip(self.events, decisions):
            if event.event_id == event_id:
                return event, decision
        return None


store = RunStore()

app = FastAPI(
    title="Arbiter API",
    version=__version__,
    description="A Gemma 4 reasoning layer over production ML drift telemetry.",
)

# The dashboard is served from a different origin in both dev and deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv("ARBITER_CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]
    + ([".vercel.app"] if os.getenv("ARBITER_ALLOW_VERCEL") else []),
    allow_origin_regex=r"https://.*\.vercel\.app" if os.getenv("ARBITER_ALLOW_VERCEL") else None,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# The SDK-compatible wire protocol (/models/register, /predict/{id}, /retrain/{id}).
# Mounted at the root because the SDK builds those paths itself from
# DRIFTGUARD_API_URL and cannot be told to use a prefix.
app.include_router(sdk_router)


# ---- request models ------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class ReasonRequest(BaseModel):
    """A caller-supplied drift event. feature_scores must be NAME-keyed."""

    model_id: str
    global_drift_score: float
    feature_scores: dict[str, float]
    drift_detected: bool = True
    reference_stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    current_stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


# ---- serialization -------------------------------------------------------


def _card(event: DriftEvent, decision: Decision) -> dict[str, Any]:
    """Triage-feed card: one number, one sentence explaining it."""
    v = decision.verdict
    return {
        "event_id": event.event_id,
        "model_id": event.model_id,
        "timestamp": event.timestamp,
        "global_drift_score": event.global_drift_score,
        "action": v.action,
        "confidence": v.confidence,
        "taxonomy": v.taxonomy,
        "reasoning": v.reasoning,
        "primary_features": v.primary_features,
        "top_features": [{"name": n, "score": s} for n, s in event.top_features(3)],
        "suppressed": v.suppressed,
        # The most persuasive element in the UI: the page that did not happen.
        "driftguard_would_page": decision.driftguard_would_page,
        "needs_human": decision.needs_human,
        "source": v.source,
    }


# ---- endpoints -----------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    client = get_client()
    return {
        "status": "ok",
        "version": __version__,
        "gemma": client.stats(),
        "replay_size": len(store.events),
        "reasoned": store._reasoned,
        "confidence_floor": CONFIDENCE_FLOOR,
        "driftguard_threshold": DRIFTGUARD_THRESHOLD,
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    """The metric row. Measured from the current run (AGENTS.md §2)."""
    decisions = store.ensure_reasoned()
    return {
        **measure_suppression(decisions),
        "accuracy": measure_accuracy(decisions, store.events),
        "corpus": summarize(store.events),
        "gemma": get_client().stats(),
    }


@app.get("/api/events")
def events() -> dict[str, Any]:
    decisions = store.ensure_reasoned()
    cards = [_card(e, d) for e, d in zip(store.events, decisions)]
    # Act-worthy first, then escalations, then suppressed — triage order.
    cards.sort(key=lambda c: (c["suppressed"], not c["needs_human"], -c["confidence"]))
    return {"events": cards, "count": len(cards)}


@app.get("/api/events/{event_id}")
def event_detail(event_id: str) -> dict[str, Any]:
    found = store.find(event_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"No event {event_id}")
    event, decision = found

    features = [
        {
            "name": name,
            "score": score,
            **event.delta(name),
        }
        for name, score in sorted(event.feature_scores.items(), key=lambda kv: -kv[1])
    ]

    body: dict[str, Any] = {
        **_card(event, decision),
        "features": features,
        "context": event.context,
        "analysis": decision.analysis.to_dict() if decision.analysis else None,
        "quality_bar": (
            meets_quality_bar(decision.analysis, event) if decision.analysis else None
        ),
    }

    # UC 6 — only meaningful once a retrain was recommended.
    if decision.verdict.action == "RETRAIN":
        report = ChallengerReport(
            model_id=event.model_id,
            champion_metric=0.9120,
            challenger_metric=0.9150,
            champion_segment_metric=0.8740,
            challenger_segment_metric=0.8530,
            segment_name=f"{decision.verdict.primary_features[0] if decision.verdict.primary_features else 'drifted'}_segment",
        )
        body["challenger"] = judge_challenger(report, event).to_dict()

    return body


@app.post("/api/ask")
def api_ask(req: AskRequest) -> dict[str, Any]:
    decisions = store.ensure_reasoned()
    return ask(req.question, decisions).to_dict()


@app.get("/api/ask/examples")
def ask_examples() -> dict[str, Any]:
    return {"examples": EXAMPLE_QUESTIONS}


@app.post("/api/reason")
def api_reason(req: ReasonRequest) -> dict[str, Any]:
    """Reason over a caller-supplied event — the SDK integration path."""
    event = DriftEvent(
        model_id=req.model_id,
        global_drift_score=req.global_drift_score,
        feature_scores=req.feature_scores,
        drift_detected=req.drift_detected,
        reference_stats=req.reference_stats,
        current_stats=req.current_stats,
        context=req.context,
    )
    client = get_client()
    analysis = analyze(event, client)
    decision = decide(event, client, analysis=analysis)
    return {
        "analysis": analysis.to_dict(),
        "verdict": decision.verdict.to_dict(),
        "needs_human": decision.needs_human,
        "quality_bar": meets_quality_bar(analysis, event),
    }


@app.post("/api/replay/reset")
def replay_reset(size: int | None = None) -> dict[str, Any]:
    """Recompute the run. Used to pre-warm the cache before presenting."""
    store.reset(size)
    decisions = store.ensure_reasoned()
    return {"replay_size": len(store.events), **measure_suppression(decisions)}


# ---- API keys ------------------------------------------------------------


class CreateKeyRequest(BaseModel):
    label: str = Field(default="default", max_length=80)


@app.post("/api/keys")
def create_api_key(req: CreateKeyRequest) -> dict[str, Any]:
    """Mint an Arbiter API key.

    The plaintext is in this response and nowhere else — it is stored only as
    a salted hash, so it cannot be shown again.
    """
    created = keys.create_key(req.label)
    return {
        **created,
        "warning": "Copy this key now. It cannot be retrieved again.",
        "setup": {
            "DRIFTGUARD_API_URL": os.getenv("ARBITER_PUBLIC_URL", "http://localhost:8000"),
            "DRIFTGUARD_API_KEY": created["key"],
        },
    }


@app.get("/api/keys")
def get_api_keys() -> dict[str, Any]:
    """List keys by hint. Plaintext is never returned."""
    return {"keys": keys.list_keys()}


@app.delete("/api/keys/{key_id}")
def delete_api_key(key_id: int) -> dict[str, Any]:
    if not keys.revoke_key(key_id):
        raise HTTPException(status_code=404, detail=f"No key {key_id}")
    return {"status": "revoked", "id": key_id}


# ---- live models and telemetry ------------------------------------------


@app.get("/api/models")
def get_models() -> dict[str, Any]:
    """Models that have registered with Arbiter via the SDK."""
    models = keys.list_models()
    for m in models:
        m["telemetry_count"] = keys.telemetry_count(m["model_id"])
        m["event_count"] = len(keys.list_events(m["model_id"], limit=500))
    return {"models": models, "count": len(models)}


@app.get("/api/telemetry")
def get_telemetry(model_id: str | None = None, limit: int = 100) -> dict[str, Any]:
    """Raw telemetry as received from the SDK."""
    return {
        "telemetry": keys.recent_telemetry(model_id, min(limit, 1000)),
        "total": keys.telemetry_count(model_id),
    }


@app.get("/api/live/events")
def get_live_events(model_id: str | None = None, limit: int = 100) -> dict[str, Any]:
    """Drift events Arbiter reasoned about from real SDK telemetry.

    Distinct from /api/events, which serves the synthetic demo corpus.
    """
    rows = keys.list_events(model_id, min(limit, 500))
    cards = []
    for r in rows:
        v = r.get("verdict_json") or {}
        e = r.get("event_json") or {}
        scores = e.get("feature_scores") or {}
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        cards.append(
            {
                "event_id": r["event_id"],
                "model_id": r["model_id"],
                "timestamp": r["created_at"],
                "global_drift_score": r.get("drift_score"),
                "action": v.get("action"),
                "confidence": v.get("confidence"),
                "taxonomy": v.get("taxonomy"),
                "reasoning": v.get("reasoning"),
                "primary_features": v.get("primary_features", []),
                "top_features": [{"name": n, "score": s} for n, s in top],
                "suppressed": v.get("suppressed", False),
                "driftguard_would_page": v.get("suppressed", False),
                "needs_human": (v.get("confidence") or 0) < CONFIDENCE_FLOOR
                or v.get("action") == "INVESTIGATE",
                "source": v.get("source"),
                "live": True,
            }
        )
    return {"events": cards, "count": len(cards)}


@app.get("/api/live/events/{event_id}")
def get_live_event_detail(event_id: str) -> dict[str, Any]:
    for r in keys.list_events(limit=500):
        if r["event_id"] == event_id:
            e = r.get("event_json") or {}
            scores = e.get("feature_scores") or {}
            ref = e.get("reference_stats") or {}
            cur = e.get("current_stats") or {}
            features = []
            for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
                rm = float((ref.get(name) or {}).get("mean", 0.0))
                cm = float((cur.get(name) or {}).get("mean", 0.0))
                features.append(
                    {
                        "name": name,
                        "score": score,
                        "ref_mean": rm,
                        "cur_mean": cm,
                        "ref_std": float((ref.get(name) or {}).get("std", 0.0)),
                        "cur_std": float((cur.get(name) or {}).get("std", 0.0)),
                        "ref_null_rate": float((ref.get(name) or {}).get("null_rate", 0.0)),
                        "cur_null_rate": float((cur.get(name) or {}).get("null_rate", 0.0)),
                        "pct_change": round((cm - rm) / abs(rm) * 100, 1) if rm else 0.0,
                    }
                )
            return {
                "event_id": r["event_id"],
                "model_id": r["model_id"],
                "timestamp": r["created_at"],
                "global_drift_score": r.get("drift_score"),
                "features": features,
                "context": e.get("context", {}),
                "verdict": r.get("verdict_json"),
                "analysis": r.get("analysis_json"),
                "live": True,
            }
    raise HTTPException(status_code=404, detail=f"No live event {event_id}")


@app.get("/api/digest")
def digest() -> dict[str, Any]:
    """The exact telemetry digest handed to Gemma for UC 11 — auditable."""
    return {"digest": build_digest(store.ensure_reasoned())}
