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

from . import __version__
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
from .query import EXAMPLE_QUESTIONS, ask, build_digest
from .reasoner import analyze, meets_quality_bar
from .scenarios import generate_replay, summarize
from .schema import ChallengerReport, DriftEvent

REPLAY_SIZE = int(os.getenv("DRIFTREASON_REPLAY_SIZE", "20"))


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
    title="DriftReason API",
    version=__version__,
    description="A Gemma 4 reasoning layer over production ML drift telemetry.",
)

# The dashboard is served from a different origin in both dev and deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv("DRIFTREASON_CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]
    + ([".vercel.app"] if os.getenv("DRIFTREASON_ALLOW_VERCEL") else []),
    allow_origin_regex=r"https://.*\.vercel\.app" if os.getenv("DRIFTREASON_ALLOW_VERCEL") else None,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


@app.get("/api/digest")
def digest() -> dict[str, Any]:
    """The exact telemetry digest handed to Gemma for UC 11 — auditable."""
    return {"digest": build_digest(store.ensure_reasoned())}
