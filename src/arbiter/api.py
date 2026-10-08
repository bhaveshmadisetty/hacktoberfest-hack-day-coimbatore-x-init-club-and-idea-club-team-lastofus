"""FastAPI layer — what the dashboard talks to.

Three groups of routes:

- ``/auth/*``   accounts and sessions
- ``/api/*``    the dashboard's own API, session-authenticated and user-scoped
- root paths    the DriftGuard SDK wire protocol (see ingest.py), API-key
                authenticated

Everything under ``/api`` that touches user data resolves the session first and
filters by ``user_id``, so one account can never read another's telemetry. The
demo corpus is the exception: it is synthetic, shared, and read-only.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__, auth, keys
from .chat import (
    FLEET_SUGGESTIONS,
    MODEL_SUGGESTIONS,
    ChatTurn,
    ask_fleet,
    ask_model,
)
from .evaluator import (
    CONFIDENCE_FLOOR,
    DRIFTGUARD_THRESHOLD,
    Decision,
    decide,
    judge_challenger,
    measure_accuracy,
    measure_suppression,
)
from .docgen import generate as generate_docs
from .gemma_client import get_client
from .ingest import router as sdk_router
from .query import EXAMPLE_QUESTIONS, ask, build_digest
from .reasoner import analyze, meets_quality_bar
from .scenarios import generate_replay, summarize
from .schema import ChallengerReport, DriftEvent

REPLAY_SIZE = int(os.getenv("ARBITER_REPLAY_SIZE", "20"))

# Auth tables and the ownership columns on pre-existing tables.
auth.ensure_auth_schema()


class RunStore:
    """The synthetic demo corpus and its decisions.

    Shared and read-only: it is the same 20 seeded events for everyone, so it
    carries no user scoping. Reasoning is computed once, lazily, then reused —
    a demo must not re-call a hosted model on every page load, and the metric
    row has to agree with the cards beneath it.
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
    description="AI judgment for production ML, powered by Gemma 4.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv("ARBITER_CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ],
    allow_origin_regex=(
        r"https://.*\.vercel\.app" if os.getenv("ARBITER_ALLOW_VERCEL") else None
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)

app.include_router(sdk_router)


# ---- session dependency --------------------------------------------------


def current_user(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Resolve the bearer session token. 401 when absent or expired."""
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    user = auth.resolve_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user


def optional_user(
    authorization: str | None = Header(default=None),
) -> dict[str, Any] | None:
    """Resolve a session if present, else None. For shared/demo routes."""
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    return auth.resolve_session(token)


# ---- request models ------------------------------------------------------


class RegisterBody(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=200)
    name: str | None = Field(default=None, max_length=120)


class LoginBody(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=200)


class ChangeEmailBody(BaseModel):
    new_email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=200)


class ChangePasswordBody(BaseModel):
    current_password: str = Field(..., max_length=200)
    new_password: str = Field(..., max_length=200)


class CreateKeyBody(BaseModel):
    label: str = Field(default="default", max_length=80)


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class ChatBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[dict[str, str]] = Field(default_factory=list)


class ReasonBody(BaseModel):
    """A caller-supplied drift event. feature_scores must be NAME-keyed."""

    model_id: str
    global_drift_score: float
    feature_scores: dict[str, float]
    drift_detected: bool = True
    reference_stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    current_stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


# ---- auth ---------------------------------------------------------------


@app.post("/auth/register")
def auth_register(body: RegisterBody) -> dict[str, Any]:
    try:
        auth.register(body.email, body.password, body.name)
        session = auth.login(body.email, body.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return session


@app.post("/auth/login")
def auth_login(body: LoginBody) -> dict[str, Any]:
    try:
        return auth.login(body.email, body.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.post("/auth/logout")
def auth_logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if authorization and authorization.lower().startswith("bearer "):
        auth.end_session(authorization[7:].strip())
    return {"status": "signed out"}


@app.get("/auth/me")
def auth_me(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    full = auth.get_user(user["id"]) or user
    return {"user": full}


@app.patch("/auth/email")
def auth_change_email(
    body: ChangeEmailBody, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    try:
        return {"user": auth.change_email(user["id"], body.new_email, body.password)}
    except auth.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/auth/password")
def auth_change_password(
    body: ChangePasswordBody, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    try:
        auth.change_password(user["id"], body.current_password, body.new_password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Every session was invalidated, so the caller must sign in again.
    return {"status": "password changed", "reauth_required": True}


# ---- health -------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    client = get_client()
    return {
        "status": "ok",
        "version": __version__,
        "gemma": client.stats(),
        "replay_size": len(store.events),
        "confidence_floor": CONFIDENCE_FLOOR,
        "driftguard_threshold": DRIFTGUARD_THRESHOLD,
        "has_users": _has_users(),
    }


def _has_users() -> bool:
    with keys.cursor() as cur:
        try:
            cur.execute("SELECT 1 FROM users LIMIT 1")
            return cur.fetchone() is not None
        except Exception:  # noqa: BLE001 - table not yet created
            return False


# ---- API keys -----------------------------------------------------------


@app.post("/api/keys")
def create_api_key(
    body: CreateKeyBody, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    """Mint an API key for the signed-in account.

    The plaintext is in this response and nowhere else — only a salted hash is
    stored, so it cannot be shown again.
    """
    created = keys.create_key(body.label, user_id=user["id"])
    return {
        **created,
        "warning": "Copy this key now. It cannot be retrieved again.",
        "setup": {
            "DRIFTGUARD_API_URL": os.getenv("ARBITER_PUBLIC_URL", "http://localhost:8000"),
            "DRIFTGUARD_API_KEY": created["key"],
        },
    }


@app.get("/api/keys")
def get_api_keys(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    return {"keys": keys.list_keys(user_id=user["id"])}


@app.delete("/api/keys/{key_id}")
def delete_api_key(
    key_id: int, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    if not keys.revoke_key(key_id, user_id=user["id"]):
        raise HTTPException(status_code=404, detail=f"No key {key_id}")
    return {"status": "revoked", "id": key_id}


# ---- the user's models --------------------------------------------------


@app.get("/api/models")
def get_models(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    """Models that registered with Arbiter using this account's keys."""
    models = keys.list_models(user_id=user["id"])
    for m in models:
        mid = m["model_id"]
        m["telemetry_count"] = keys.telemetry_count(mid, user_id=user["id"])
        events = keys.list_events(mid, limit=500, user_id=user["id"])
        m["event_count"] = len(events)
        acted = 0
        suppressed = 0
        for e in events:
            v = e.get("verdict_json") or {}
            if v.get("action") in ("RETRAIN", "ROLLBACK"):
                acted += 1
            else:
                suppressed += 1
        m["actionable"] = acted
        m["suppressed"] = suppressed
        latest = events[0] if events else None
        m["latest_verdict"] = (latest.get("verdict_json") if latest else None)
        m["latest_drift_score"] = (latest.get("drift_score") if latest else None)
    return {"models": models, "count": len(models)}


@app.get("/api/models/{model_id}/metrics")
def model_metrics(
    model_id: str, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    """Per-model metric row. Measured from that model's own events."""
    model = keys.get_model(model_id)
    if not model or model.get("user_id") not in (None, user["id"]):
        raise HTTPException(status_code=404, detail=f"No model {model_id}")

    events = keys.list_events(model_id, limit=500, user_id=user["id"])
    total = len(events)
    by_action: dict[str, int] = {}
    by_taxonomy: dict[str, int] = {}
    gemma_backed = 0
    escalated = 0
    confidences: list[float] = []

    for e in events:
        v = e.get("verdict_json") or {}
        action = v.get("action") or "UNKNOWN"
        by_action[action] = by_action.get(action, 0) + 1
        tax = v.get("taxonomy") or "undetermined"
        by_taxonomy[tax] = by_taxonomy.get(tax, 0) + 1
        if str(v.get("source", "")).startswith("gemma"):
            gemma_backed += 1
        conf = v.get("confidence")
        if isinstance(conf, (int, float)):
            confidences.append(float(conf))
            if conf < CONFIDENCE_FLOOR:
                escalated += 1

    actionable = by_action.get("RETRAIN", 0) + by_action.get("ROLLBACK", 0)
    telemetry = keys.recent_telemetry(model_id, limit=200, user_id=user["id"])
    scores = [r["drift_score"] for r in telemetry if r.get("drift_score") is not None]

    return {
        "model_id": model_id,
        "features": model.get("features") or [],
        "drift_threshold": model.get("drift_threshold") or DRIFTGUARD_THRESHOLD,
        "version": model.get("version"),
        "reported_accuracy": model.get("accuracy"),
        "telemetry_count": keys.telemetry_count(model_id, user_id=user["id"]),
        "alerts_fired": total,
        "actionable": actionable,
        "suppressed": total - actionable,
        "noise_reduction_pct": round((total - actionable) / total * 100, 1) if total else 0.0,
        "escalated_to_human": escalated,
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "by_action": dict(sorted(by_action.items())),
        "by_taxonomy": dict(sorted(by_taxonomy.items())),
        "gemma_backed_decisions": gemma_backed,
        # AGENTS.md §2 — a run served by the fallback measures the fallback.
        "reportable": total > 0 and gemma_backed == total,
        "drift_series": [
            {"drift_score": r["drift_score"], "received_at": r["received_at"]}
            for r in reversed(telemetry)
            if r.get("drift_score") is not None
        ],
        "latest_drift_score": scores[0] if scores else None,
    }


@app.get("/api/models/{model_id}/events")
def model_events(
    model_id: str, user: dict[str, Any] = Depends(current_user), limit: int = 100
) -> dict[str, Any]:
    rows = keys.list_events(model_id, min(limit, 500), user_id=user["id"])
    return {"events": [_live_card(r) for r in rows], "count": len(rows)}


@app.get("/api/models/{model_id}/telemetry")
def model_telemetry(
    model_id: str, user: dict[str, Any] = Depends(current_user), limit: int = 100
) -> dict[str, Any]:
    return {
        "telemetry": keys.recent_telemetry(model_id, min(limit, 1000), user_id=user["id"]),
        "total": keys.telemetry_count(model_id, user_id=user["id"]),
    }


# ---- chat ---------------------------------------------------------------


@app.get("/api/models/{model_id}/docs")
def model_docs(
    model_id: str, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    """Gemma-written incident documentation for one model.

    Built from stored telemetry and verdicts only — the incident list is
    assembled server-side and handed to the model, so the report cannot
    contain an incident that did not happen.
    """
    model = keys.get_model(model_id)
    if not model or model.get("user_id") not in (None, user["id"]):
        raise HTTPException(status_code=404, detail=f"No model {model_id}")
    return generate_docs(model_id, user_id=user["id"]).to_dict()


@app.post("/api/models/{model_id}/chat")
def chat_about_model(
    model_id: str, body: ChatBody, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    """Chat scoped to one model, with that model's telemetry as context."""
    model = keys.get_model(model_id)
    if not model or model.get("user_id") not in (None, user["id"]):
        raise HTTPException(status_code=404, detail=f"No model {model_id}")

    history = [
        ChatTurn(role=str(t.get("role", "user")), content=str(t.get("content", "")))
        for t in body.history[-10:]
    ]
    return ask_model(model_id, body.question, history, user_id=user["id"]).to_dict()


@app.post("/api/chat")
def chat_about_fleet(
    body: ChatBody, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    """Fleet-wide chat, with one summary line per model as context."""
    history = [
        ChatTurn(role=str(t.get("role", "user")), content=str(t.get("content", "")))
        for t in body.history[-10:]
    ]
    return ask_fleet(body.question, history, user_id=user["id"]).to_dict()


@app.get("/api/chat/suggestions")
def chat_suggestions(model_id: str | None = None) -> dict[str, Any]:
    return {"suggestions": MODEL_SUGGESTIONS if model_id else FLEET_SUGGESTIONS}


# ---- live events --------------------------------------------------------


def _live_card(row: dict[str, Any]) -> dict[str, Any]:
    v = row.get("verdict_json") or {}
    e = row.get("event_json") or {}
    scores = e.get("feature_scores") or {}
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
    conf = v.get("confidence") or 0
    return {
        "event_id": row["event_id"],
        "model_id": row["model_id"],
        "timestamp": row["created_at"],
        "global_drift_score": row.get("drift_score"),
        "action": v.get("action"),
        "confidence": conf,
        "taxonomy": v.get("taxonomy"),
        "reasoning": v.get("reasoning"),
        "primary_features": v.get("primary_features", []),
        "top_features": [{"name": n, "score": s} for n, s in top],
        "suppressed": v.get("suppressed", False),
        "driftguard_would_page": v.get("suppressed", False),
        "needs_human": conf < CONFIDENCE_FLOOR or v.get("action") == "INVESTIGATE",
        "source": v.get("source"),
        "live": True,
    }


@app.get("/api/live/events")
def get_live_events(
    user: dict[str, Any] = Depends(current_user),
    model_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    rows = keys.list_events(model_id, min(limit, 500), user_id=user["id"])
    return {"events": [_live_card(r) for r in rows], "count": len(rows)}


@app.get("/api/live/events/{event_id}")
def get_live_event_detail(
    event_id: str, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    for r in keys.list_events(limit=500, user_id=user["id"]):
        if r["event_id"] != event_id:
            continue
        e = r.get("event_json") or {}
        scores = e.get("feature_scores") or {}
        ref = e.get("reference_stats") or {}
        cur = e.get("current_stats") or {}
        features = []
        for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            rs, cs = ref.get(name) or {}, cur.get(name) or {}
            rm, cm = float(rs.get("mean", 0.0)), float(cs.get("mean", 0.0))
            features.append(
                {
                    "name": name,
                    "score": score,
                    "ref_mean": rm,
                    "cur_mean": cm,
                    "ref_std": float(rs.get("std", 0.0)),
                    "cur_std": float(cs.get("std", 0.0)),
                    "ref_null_rate": float(rs.get("null_rate", 0.0)),
                    "cur_null_rate": float(cs.get("null_rate", 0.0)),
                    "pct_change": round((cm - rm) / abs(rm) * 100, 1) if rm else 0.0,
                }
            )
        return {
            **_live_card(r),
            "features": features,
            "context": e.get("context", {}),
            "analysis": r.get("analysis_json"),
            "quality_bar": None,
        }
    raise HTTPException(status_code=404, detail=f"No live event {event_id}")


# ---- demo corpus (synthetic, shared, read-only) -------------------------


def _card(event: DriftEvent, decision: Decision) -> dict[str, Any]:
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
        "driftguard_would_page": decision.driftguard_would_page,
        "needs_human": decision.needs_human,
        "source": v.source,
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    """Demo-corpus metric row. Measured from the current run (AGENTS.md §2)."""
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
    cards.sort(key=lambda c: (c["suppressed"], not c["needs_human"], -c["confidence"]))
    return {"events": cards, "count": len(cards)}


@app.get("/api/events/{event_id}")
def event_detail(event_id: str) -> dict[str, Any]:
    found = store.find(event_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"No event {event_id}")
    event, decision = found

    features = [
        {"name": name, "score": score, **event.delta(name)}
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

    if decision.verdict.action == "RETRAIN":
        primary = (
            decision.verdict.primary_features[0]
            if decision.verdict.primary_features
            else "drifted"
        )
        report = ChallengerReport(
            model_id=event.model_id,
            champion_metric=0.9120,
            challenger_metric=0.9150,
            champion_segment_metric=0.8740,
            challenger_segment_metric=0.8530,
            segment_name=f"{primary}_segment",
        )
        body["challenger"] = judge_challenger(report, event).to_dict()

    return body


@app.post("/api/ask")
def api_ask(body: AskBody) -> dict[str, Any]:
    return ask(body.question, store.ensure_reasoned()).to_dict()


@app.get("/api/ask/examples")
def ask_examples() -> dict[str, Any]:
    return {"examples": EXAMPLE_QUESTIONS}


@app.post("/api/reason")
def api_reason(body: ReasonBody) -> dict[str, Any]:
    """Reason over a caller-supplied event — the SDK integration path."""
    event = DriftEvent(
        model_id=body.model_id,
        global_drift_score=body.global_drift_score,
        feature_scores=body.feature_scores,
        drift_detected=body.drift_detected,
        reference_stats=body.reference_stats,
        current_stats=body.current_stats,
        context=body.context,
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
    """Recompute the demo run. Used to pre-warm the cache before presenting."""
    store.reset(size)
    decisions = store.ensure_reasoned()
    return {"replay_size": len(store.events), **measure_suppression(decisions)}


@app.get("/api/digest")
def digest() -> dict[str, Any]:
    """The exact telemetry digest handed to Gemma for UC 11 — auditable."""
    return {"digest": build_digest(store.ensure_reasoned())}
