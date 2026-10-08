"""Chatbots over drift telemetry — one per model, plus a fleet-wide one.

Two scopes, and the difference is the context each is given:

- **Model chat** sees one model's feature names, its drift events, its
  verdicts and its recent telemetry statistics. It can answer
  "why did transaction_amount drift?" because those numbers are in the prompt.
- **Fleet chat** sees every model the user owns, one summary line each. It can
  answer "which model is worst?" but deliberately cannot quote per-feature
  magnitudes for a specific model — that context was not given to it, and the
  system prompt tells it to say so rather than guess.

Both refuse to invent. The answer comes only from the context block, and when
the context does not contain it the model is instructed to say exactly that.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from . import keys
from .gemma_client import GemmaClient, get_client

SYSTEM_MODEL_CHAT = """You are Arbiter's analyst for one specific machine-learning model in production. You answer questions about that model's drift, its features, its verdicts, and what its operator should do next.

Rules you always follow:
- Answer only from the telemetry context provided. If the answer is not in it, say so plainly in one sentence. Never estimate, extrapolate, or invent a number.
- Cite specific feature names and specific numeric magnitudes from the context (e.g. "transaction_amount moved from a reference mean of 240 to 890, +271%").
- When asked about cause, distinguish a data-pipeline failure (nulls appear, variance collapses toward a default) from a genuine population shift (internally consistent values, no nulls) from model decay (inputs stable, performance falling).
- When asked what to do, be concrete and name the action: retrain, roll back, investigate upstream, or do nothing.
- If asked about code, give a short, correct snippet using the DriftGuard SDK against Arbiter.
- Be direct. 2-6 sentences of prose unless a list is genuinely clearer. No markdown headings."""


SYSTEM_FLEET_CHAT = """You are Arbiter's analyst for a fleet of machine-learning models in production. You answer questions that span models: which are degrading, which share a cause, where an operator should look first.

Rules you always follow:
- Answer only from the fleet context provided. If the answer is not in it, say so plainly. Never invent a model, a feature name, or a number.
- You are given one summary line per model, not full per-feature telemetry. If a question needs per-feature detail for one model, answer what you can from the summary and tell the user to open that model's own chat for feature-level analysis.
- When several models drift at once, consider whether a shared upstream dependency is the more likely cause than several independent coincidences — but only say so if the context supports it.
- Be direct. 2-6 sentences of prose. No markdown headings."""


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ChatReply:
    answer: str
    scope: str  # "model" | "fleet"
    model_id: str | None = None
    context_events: int = 0
    source: str = "gemma"
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "scope": self.scope,
            "model_id": self.model_id,
            "context_events": self.context_events,
            "source": self.source,
            "suggestions": self.suggestions,
        }


# Seeded prompts, so a chat panel never starts empty.
MODEL_SUGGESTIONS = [
    "Why did this model drift?",
    "Should I retrain it?",
    "Which feature is the biggest problem?",
    "Is this a pipeline bug or a real population shift?",
]

FLEET_SUGGESTIONS = [
    "Which models degraded and why?",
    "Are any alerts being suppressed that I should look at?",
    "Do any of these models share a root cause?",
    "Where should I look first?",
]


def _fmt(x: float) -> str:
    """Adaptive precision — a ratio feature printed as 0.3 hides its movement."""
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def build_model_context(model_id: str, user_id: int | None = None) -> tuple[str, int]:
    """Context block for one model. Returns (context, events_included)."""
    model = keys.get_model(model_id)
    if not model:
        return f"No model named {model_id} is registered with Arbiter.", 0

    names: list[str] = model.get("features") or []
    lines = [
        f"MODEL: {model_id}",
        f"  version          : {model.get('version') or 'unknown'}",
        f"  drift threshold  : {model.get('drift_threshold') or 0.15}",
        f"  reported accuracy: {model.get('accuracy') if model.get('accuracy') is not None else 'not reported'}",
        f"  registered       : {model.get('registered_at')}",
        f"  last seen        : {model.get('last_seen') or 'never'}",
        f"  features ({len(names)}): {', '.join(names) if names else 'NONE REGISTERED — narrations cannot name features'}",
        f"  telemetry rows   : {keys.telemetry_count(model_id, user_id=user_id)}",
    ]

    # Recent telemetry statistics, so feature-level questions are answerable
    # even before a drift event has been reasoned about.
    rows = keys.recent_telemetry(model_id, limit=200, user_id=user_id)
    vectors = [r["features"] for r in rows if isinstance(r.get("features"), list)]
    if vectors and names:
        lines.append("")
        lines.append("RECENT FEATURE STATISTICS (newest half vs. older half):")
        half = max(1, len(vectors) // 2)
        recent, older = vectors[:half], vectors[half:] or vectors[:half]
        for i, name in enumerate(names):
            cur = [float(v[i]) for v in recent if i < len(v) and v[i] is not None]
            ref = [float(v[i]) for v in older if i < len(v) and v[i] is not None]
            if not cur or not ref:
                continue
            cm, rm = statistics.fmean(cur), statistics.fmean(ref)
            pct = ((cm - rm) / abs(rm) * 100) if rm else 0.0
            cs = statistics.pstdev(cur) if len(cur) > 1 else 0.0
            rs = statistics.pstdev(ref) if len(ref) > 1 else 0.0
            lines.append(
                f"  {name:<22} ref mean {_fmt(rm)} std {_fmt(rs)} -> "
                f"cur mean {_fmt(cm)} std {_fmt(cs)} ({pct:+.1f}%)"
            )

    drift_scores = [r["drift_score"] for r in rows if r.get("drift_score") is not None]
    if drift_scores:
        lines.append("")
        lines.append(
            f"DRIFT SCORES (last {len(drift_scores)} predictions): "
            f"latest {drift_scores[0]:.3f}, mean {statistics.fmean(drift_scores):.3f}, "
            f"max {max(drift_scores):.3f}"
        )

    events = keys.list_events(model_id, limit=10, user_id=user_id)
    if events:
        lines.append("")
        lines.append(f"DRIFT EVENTS ARBITER REASONED ABOUT ({len(events)}):")
        for e in events:
            v = e.get("verdict_json") or {}
            a = e.get("analysis_json") or {}
            lines.append(
                f"  {e['event_id']} ({e['created_at']}) drift {e.get('drift_score')} "
                f"-> {v.get('action')} at {v.get('confidence')} confidence, "
                f"taxonomy {v.get('taxonomy')}"
            )
            if v.get("reasoning"):
                lines.append(f"    verdict: {v['reasoning']}")
            if a.get("hypothesis"):
                lines.append(f"    root cause: {a['hypothesis']}")
    else:
        lines.append("")
        lines.append(
            "DRIFT EVENTS: none yet. Either drift has not crossed the threshold, "
            "or not enough telemetry has accumulated to reason about it."
        )

    return "\n".join(lines), len(events)


def build_fleet_context(user_id: int | None = None) -> tuple[str, int]:
    """Context block covering every model the user owns."""
    models = keys.list_models(user_id=user_id)
    if not models:
        return (
            "No models are connected to Arbiter yet. The user needs to generate an "
            "API key and point the DriftGuard SDK at Arbiter.",
            0,
        )

    lines = [f"FLEET: {len(models)} model(s) connected to Arbiter.", ""]
    total_events = 0

    for m in models:
        mid = m["model_id"]
        events = keys.list_events(mid, limit=5, user_id=user_id)
        total_events += len(events)
        n_tel = keys.telemetry_count(mid, user_id=user_id)

        lines.append(f"MODEL {mid}")
        lines.append(
            f"  features: {', '.join(m.get('features') or []) or 'none registered'}"
        )
        lines.append(
            f"  telemetry rows: {n_tel} | drift threshold: {m.get('drift_threshold') or 0.15} "
            f"| last seen: {m.get('last_seen') or 'never'}"
        )

        if events:
            for e in events:
                v = e.get("verdict_json") or {}
                lines.append(
                    f"  event {e['event_id']}: drift {e.get('drift_score')} -> "
                    f"{v.get('action')} ({v.get('confidence')}) {v.get('taxonomy')}"
                )
                if v.get("reasoning"):
                    lines.append(f"    {v['reasoning']}")
        else:
            lines.append("  no drift events reasoned about yet")
        lines.append("")

    return "\n".join(lines), total_events


def _history_block(history: list[ChatTurn], limit: int = 6) -> str:
    """Recent turns, so follow-ups like "and the second one?" resolve."""
    if not history:
        return ""
    recent = history[-limit:]
    out = ["", "CONVERSATION SO FAR:"]
    for turn in recent:
        who = "User" if turn.role == "user" else "You"
        out.append(f"  {who}: {turn.content.strip()[:400]}")
    return "\n".join(out)


def ask_model(
    model_id: str,
    question: str,
    history: list[ChatTurn] | None = None,
    user_id: int | None = None,
    client: GemmaClient | None = None,
) -> ChatReply:
    """Answer a question about one specific model."""
    question = (question or "").strip()
    if not question:
        return ChatReply(
            answer="Ask about this model's drift, its features, or what to do next.",
            scope="model",
            model_id=model_id,
            source="static",
            suggestions=MODEL_SUGGESTIONS,
        )

    context, n_events = build_model_context(model_id, user_id)
    client = client or get_client()

    prompt = f"""Telemetry context for the model this conversation is about:

{context}
{_history_block(history or [])}

User's question: {question}

Answer from the context above only. Cite specific feature names and numbers where the context provides them. If the context does not contain what is needed, say so in one sentence rather than estimating."""

    text = client.generate(prompt, SYSTEM_MODEL_CHAT).strip()

    if not text:
        return ChatReply(
            answer=(
                "Gemma is unavailable right now, so I cannot reason about this model. "
                f"What I can tell you from stored telemetry: {n_events} drift event(s) "
                f"reasoned about, "
                f"{keys.telemetry_count(model_id, user_id=user_id)} telemetry rows received."
            ),
            scope="model",
            model_id=model_id,
            context_events=n_events,
            source="fallback",
            suggestions=MODEL_SUGGESTIONS,
        )

    return ChatReply(
        answer=text,
        scope="model",
        model_id=model_id,
        context_events=n_events,
        source=f"gemma:{client.model}",
        suggestions=MODEL_SUGGESTIONS,
    )


def ask_fleet(
    question: str,
    history: list[ChatTurn] | None = None,
    user_id: int | None = None,
    client: GemmaClient | None = None,
) -> ChatReply:
    """Answer a question spanning every model the user owns."""
    question = (question or "").strip()
    if not question:
        return ChatReply(
            answer="Ask about your models — which are drifting, why, and what to do.",
            scope="fleet",
            source="static",
            suggestions=FLEET_SUGGESTIONS,
        )

    context, n_events = build_fleet_context(user_id)
    client = client or get_client()

    prompt = f"""Fleet context — every model connected to Arbiter for this account:

{context}
{_history_block(history or [])}

User's question: {question}

Answer from the context above only. You have one summary line per model, not full per-feature telemetry: if the question needs feature-level detail for a single model, say what you can and point the user to that model's own chat. Never invent a model name, feature, or number."""

    text = client.generate(prompt, SYSTEM_FLEET_CHAT).strip()

    if not text:
        models = keys.list_models(user_id=user_id)
        return ChatReply(
            answer=(
                "Gemma is unavailable right now. From stored telemetry: "
                f"{len(models)} model(s) connected, {n_events} drift event(s) reasoned about."
            ),
            scope="fleet",
            context_events=n_events,
            source="fallback",
            suggestions=FLEET_SUGGESTIONS,
        )

    return ChatReply(
        answer=text,
        scope="fleet",
        context_events=n_events,
        source=f"gemma:{client.model}",
        suggestions=FLEET_SUGGESTIONS,
    )
