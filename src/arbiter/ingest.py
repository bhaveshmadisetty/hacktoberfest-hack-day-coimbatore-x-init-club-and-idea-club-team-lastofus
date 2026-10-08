"""SDK-compatible ingest — Arbiter as a drop-in DriftGuard backend.

The DriftGuard SDK reads its backend address from one environment variable, so
pointing it at Arbiter needs no fork and no code change:

    DRIFTGUARD_API_URL=http://localhost:8000
    DRIFTGUARD_API_KEY=ak_live_...        # minted in the Arbiter dashboard

This module implements the SDK's actual wire protocol, read off
driftguard-ai-sdk 1.0.4 rather than guessed:

    POST /models/register        tracker.py:132   X-API-Key header
    POST /predict/{model_id}     tracker.py:328   per-prediction telemetry
    GET  /models/{model_id}      tracker.py:85    version lookup
    POST /retrain/{model_id}     callback_runner.py:266
    POST /retrain/{model_id}/complete

Two details that make this work rather than merely respond:

- ``/models/register`` carries ``features: [...]`` — the ordered feature names.
  Storing them is what lets a positional drift score become
  "transaction_amount drifted", so registration is not bookkeeping here, it is
  the thing that makes reasoning possible at all.
- ``/predict`` carries ``drift_score``. When it crosses the model's threshold,
  Arbiter builds a DriftEvent from accumulated telemetry and reasons over it.
  That is the moment a plain logging backend becomes a judgment layer.
"""

from __future__ import annotations

import statistics
import threading
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import keys
from .evaluator import decide
from .gemma_client import get_client
from .schema import DriftEvent, map_feature_scores, utc_now

router = APIRouter(tags=["sdk"])

# Reasoning is slow relative to a prediction loop, so an event is reasoned
# about once and only once even if telemetry keeps streaming in.
_REASONING: set[str] = set()
_REASONING_LOCK = threading.Lock()

# Below this many samples the reference/current statistics are too noisy to
# narrate honestly, so Arbiter stores telemetry but does not yet reason.
MIN_SAMPLES_FOR_REASONING = 40


class RegisterRequest(BaseModel):
    """Mirrors tracker.py:_register_model exactly."""

    model_id: str
    project_id: int | None = None
    drift_threshold: float | None = None
    version: str | None = "1.0.0"
    accuracy: float | None = None
    features: list[str] = Field(default_factory=list)


class TelemetryRequest(BaseModel):
    """Mirrors tracker.py:_send_telemetry_async.

    ``api_key`` appears in the body as well as the header; the header wins and
    the body copy is ignored so a key is never trusted from two places.
    """

    model_id: str | None = None
    api_key: str | None = None
    features: Any = None
    prediction: Any = None
    drift_score: float | None = None


def _auth(x_api_key: str | None, *, required: bool = True) -> dict[str, Any] | None:
    """Validate the X-API-Key header.

    When no key has ever been minted, Arbiter runs open so a first-time user
    can see telemetry before they understand the key flow. Once any key
    exists, authentication is enforced — otherwise minting one would silently
    weaken the thing it was meant to secure.
    """
    record = keys.verify_key(x_api_key)
    if record:
        return record
    if not required:
        return None
    if not keys.has_any_key():
        return None
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing X-API-Key. Generate one in the Arbiter dashboard.",
    )


def _column_stats(rows: list[list[float]], names: list[str]) -> dict[str, dict[str, float]]:
    """Per-feature mean/std/null_rate over a window of feature vectors."""
    out: dict[str, dict[str, float]] = {}
    if not rows:
        return out

    width = min(len(names), max(len(r) for r in rows))
    for i in range(width):
        col = []
        nulls = 0
        for r in rows:
            if i >= len(r):
                continue
            v = r[i]
            if v is None or (isinstance(v, float) and v != v):  # None or NaN
                nulls += 1
                continue
            try:
                col.append(float(v))
            except (TypeError, ValueError):
                nulls += 1

        total = len(rows)
        out[names[i]] = {
            "mean": round(statistics.fmean(col), 4) if col else 0.0,
            "std": round(statistics.pstdev(col), 4) if len(col) > 1 else 0.0,
            "null_rate": round(nulls / total, 4) if total else 0.0,
        }
    return out


def _queue_reasoning(model_id: str, drift_score: float) -> bool:
    """Hand reasoning to a daemon thread and return immediately.

    FastAPI's BackgroundTasks run *after* the response under uvicorn but
    synchronously inside TestClient, which made a drift-crossing request take
    ~116s in tests while appearing fine in production. A real thread behaves
    the same in both, so the SDK's 5s worker timeout is never at risk.
    """
    thread = threading.Thread(
        target=_maybe_reason,
        args=(model_id, drift_score),
        name=f"arbiter-reason-{model_id}",
        daemon=True,
    )
    thread.start()
    return True


def _should_reason(model_id: str, drift_score: float) -> bool:
    """Cheap pre-check, so a background task is only queued when it will act.

    Mirrors the conditions in _maybe_reason without touching Gemma, keeping
    the hot telemetry path free of wasted task scheduling.
    """
    model = keys.get_model(model_id)
    if not model or not model.get("features"):
        return False
    if drift_score < (model.get("drift_threshold") or 0.15):
        return False
    if keys.telemetry_count(model_id) < MIN_SAMPLES_FOR_REASONING:
        return False
    with _REASONING_LOCK:
        return not _REASONING


def _maybe_reason(model_id: str, drift_score: float) -> dict[str, Any] | None:
    """Reason about a drift event if the score warrants it.

    Splits the stored telemetry window in half: the older half is the
    reference, the newer half is current. That is what makes the narration
    specific — Gemma gets "240 -> 890", not just a score.
    """
    model = keys.get_model(model_id)
    if not model:
        return None

    names: list[str] = model.get("features") or []
    threshold = model.get("drift_threshold") or 0.15
    if drift_score < threshold or not names:
        return None

    rows = keys.recent_telemetry(model_id, limit=400)
    if len(rows) < MIN_SAMPLES_FOR_REASONING:
        return None

    event_id = f"{model_id}-{len(rows)}"
    with _REASONING_LOCK:
        if event_id in _REASONING:
            return None
        _REASONING.add(event_id)

    try:
        vectors = [r["features"] for r in rows if isinstance(r.get("features"), list)]
        if len(vectors) < MIN_SAMPLES_FOR_REASONING:
            return None

        # rows arrive newest-first
        half = len(vectors) // 2
        current_stats = _column_stats(vectors[:half], names)
        reference_stats = _column_stats(vectors[half:], names)

        # The SDK sends one global score per prediction, not per-feature
        # scores. Attribute drift per feature from how far each one moved,
        # in standard deviations of its own reference window.
        feature_scores: dict[str, float] = {}
        for name in names:
            ref = reference_stats.get(name, {})
            cur = current_stats.get(name, {})
            ref_std = ref.get("std") or 0.0
            shift = abs(cur.get("mean", 0.0) - ref.get("mean", 0.0))
            z = (shift / ref_std) if ref_std > 1e-9 else (1.0 if shift > 1e-9 else 0.0)
            null_jump = max(0.0, cur.get("null_rate", 0.0) - ref.get("null_rate", 0.0))
            feature_scores[name] = round(min(1.0, z / 3.0 + null_jump), 4)

        event = DriftEvent(
            model_id=model_id,
            global_drift_score=round(float(drift_score), 4),
            feature_scores=feature_scores,
            drift_detected=True,
            reference_stats=reference_stats,
            current_stats=current_stats,
            event_id=event_id,
            timestamp=utc_now(),
            context={
                "source": "driftguard-ai-sdk telemetry (live)",
                "samples_analysed": len(vectors),
                "model_version": model.get("version"),
                "sdk_drift_threshold": threshold,
                "reported_accuracy": model.get("accuracy"),
            },
        )

        decision = decide(event, get_client())
        keys.save_event(
            event_id=event_id,
            model_id=model_id,
            drift_score=event.global_drift_score,
            event_json=event.to_dict(),
            verdict_json=decision.verdict.to_dict(),
            analysis_json=decision.analysis.to_dict() if decision.analysis else None,
        )
        return {
            "event_id": event_id,
            "action": decision.verdict.action,
            "confidence": decision.verdict.confidence,
            "taxonomy": decision.verdict.taxonomy,
            "reasoning": decision.verdict.reasoning,
            "needs_human": decision.needs_human,
        }
    finally:
        with _REASONING_LOCK:
            _REASONING.discard(event_id)


# ---- SDK endpoints -------------------------------------------------------


@router.post("/models/register")
def register_model(
    req: RegisterRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """SDK model registration — tracker.py:132.

    The ``features`` list is the valuable part: ordered names, which is what
    makes a positional drift score narratable.
    """
    record = _auth(x_api_key)
    keys.register_model(
        model_id=req.model_id,
        features=req.features,
        key_id=record["id"] if record else None,
        drift_threshold=req.drift_threshold,
        version=req.version,
        accuracy=req.accuracy,
    )
    return {
        "status": "registered",
        "model_id": req.model_id,
        "features_registered": len(req.features),
        "reasoning_enabled": bool(req.features),
        "message": (
            f"Arbiter is reasoning over {req.model_id}."
            if req.features
            else "Registered, but no feature names were sent — narrations cannot "
            "name features. Pass a feature_extractor or features= to the SDK."
        ),
    }


@router.post("/predict/{model_id}")
def ingest_telemetry(
    model_id: str,
    req: TelemetryRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Per-prediction telemetry — tracker.py:328.

    Returns as soon as the row is stored. Reasoning takes ~25s for two Gemma
    calls, and the SDK calls this from a worker behind the prediction loop
    (tracker.py:_telemetry_worker_loop, 5s timeout, 5 retries) — doing it
    inline would time out the worker and back up its queue, dropping
    telemetry. So the event is queued and the verdict is collected from
    /api/live/events, which is how the dashboard reads it anyway.
    """
    _auth(x_api_key)

    if not keys.get_model(model_id):
        # Telemetry can arrive before registration; keep it rather than drop it.
        keys.register_model(model_id=model_id, features=[])

    keys.record_telemetry(
        model_id=model_id,
        features=req.features,
        prediction=req.prediction,
        drift_score=req.drift_score,
    )

    reasoning_queued = False
    if req.drift_score is not None and _should_reason(model_id, float(req.drift_score)):
        reasoning_queued = _queue_reasoning(model_id, float(req.drift_score))

    return {
        "status": "ok",
        "model_id": model_id,
        "reasoning_queued": reasoning_queued,
    }


@router.get("/models/{model_id}")
def get_model(
    model_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Model version lookup — tracker.py:85."""
    _auth(x_api_key, required=False)
    model = keys.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not registered")
    return {
        "model_id": model["model_id"],
        "version": model.get("version") or "1.0.0",
        "features": model.get("features", []),
        "drift_threshold": model.get("drift_threshold"),
        "accuracy": model.get("accuracy"),
        "last_seen": model.get("last_seen"),
    }


@router.post("/retrain/{model_id}")
async def request_retrain(
    model_id: str,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Retrain trigger — callback_runner.py:266.

    This is the gate. The SDK asks to retrain because a number crossed a
    constant; Arbiter answers with the most recent reasoned verdict, so a
    retrain driven by a pipeline fault is refused instead of executed.
    """
    _auth(x_api_key)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - the SDK may send an empty body
        body = {}

    recent = keys.list_events(model_id, limit=1)
    verdict = (recent[0].get("verdict_json") or {}) if recent else {}
    action = verdict.get("action")
    approved = action == "RETRAIN"

    return {
        "status": "approved" if approved else "suppressed",
        "event_id": (recent[0]["event_id"] if recent else f"{model_id}-unreasoned"),
        "model_id": model_id,
        "arbiter_action": action,
        "confidence": verdict.get("confidence"),
        "reasoning": verdict.get(
            "reasoning",
            "No reasoned verdict yet for this model — not enough telemetry.",
        ),
        "trigger": body.get("trigger"),
    }


@router.post("/retrain/{model_id}/complete")
async def complete_retrain(
    model_id: str,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Retrain completion callback — callback_runner.py:302."""
    _auth(x_api_key)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    return {"status": "recorded", "model_id": model_id, "received": bool(body)}
