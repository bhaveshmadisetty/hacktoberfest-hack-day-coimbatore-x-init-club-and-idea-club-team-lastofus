"""UC 11 — natural-language queries over drift telemetry.

"Which models degraded this week and why?" answered from the event corpus
rather than from a dashboard filter. This is the tab a judge gets to type into,
so it has to refuse to invent facts: the digest handed to Gemma is the only
source, and the system prompt says to say so when the answer is not in it.

The digest is compact on purpose. Twenty full events would crowd out the
question; one line per event plus the reasoning for the ones that mattered
keeps the relevant evidence in front of the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluator import Decision
from .gemma_client import GemmaClient, get_client
from .prompts import SYSTEM_QUERY, query_prompt

# Seeded so the tab never starts empty in a demo.
EXAMPLE_QUESTIONS = [
    "Which models degraded this week and why?",
    "Were any alerts suppressed that should not have been?",
    "Which drift events were caused by broken data pipelines rather than real change?",
]


@dataclass
class Answer:
    """Response to a telemetry question, with the events it drew on."""

    question: str
    answer: str
    events_considered: int
    source: str = "gemma"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "events_considered": self.events_considered,
            "source": self.source,
        }


def build_digest(decisions: list[Decision], max_reasoning: int = 8) -> str:
    """Render decisions as a compact telemetry digest for the model.

    Every verdict appears as a one-liner so counting questions ("how many were
    suppressed?") are answerable; the most consequential ones also carry their
    reasoning so causal questions ("why?") are too.
    """
    if not decisions:
        return "(no drift events in this window)"

    lines = ["Drift events in this window:"]
    for d in decisions:
        v = d.verdict
        tax = v.taxonomy or "undetermined"
        flag = " [SUPPRESSED]" if v.suppressed else ""
        human = " [ESCALATED]" if d.needs_human else ""
        lines.append(
            f"  {d.event_id} {d.model_id} | {v.action} (confidence {v.confidence:.2f}) "
            f"| {tax}{flag}{human}"
        )

    # Act-worthy and escalated events first: those are what questions are about.
    ranked = sorted(
        decisions,
        key=lambda d: (not d.verdict.acts, not d.needs_human, -d.verdict.confidence),
    )[:max_reasoning]

    lines.append("")
    lines.append("Reasoning for the most consequential events:")
    for d in ranked:
        feats = ", ".join(d.verdict.primary_features[:3]) or "none identified"
        lines.append(f"  {d.event_id} ({d.model_id}, features: {feats})")
        lines.append(f"    {d.verdict.reasoning}")

    return "\n".join(lines)


def ask(
    question: str,
    decisions: list[Decision],
    client: GemmaClient | None = None,
) -> Answer:
    """UC 11 — answer a natural-language question over the corpus."""
    question = (question or "").strip()
    if not question:
        return Answer(
            question=question,
            answer="Ask a question about the drift telemetry — for example: "
            + EXAMPLE_QUESTIONS[0],
            events_considered=len(decisions),
            source="static",
        )

    client = client or get_client()
    digest = build_digest(decisions)
    text = client.generate(query_prompt(question, digest), SYSTEM_QUERY).strip()

    if not text:
        # No model, no invention. A deterministic count is honest; a prose
        # answer assembled from templates would read like reasoning and is not.
        return Answer(
            question=question,
            answer=_factual_summary(decisions),
            events_considered=len(decisions),
            source="fallback",
        )

    return Answer(
        question=question,
        answer=text,
        events_considered=len(decisions),
        source=f"gemma:{client.model}",
    )


def _factual_summary(decisions: list[Decision]) -> str:
    """Counts only, stated as counts. Used when Gemma is unavailable."""
    if not decisions:
        return "No drift events were recorded in this window."

    total = len(decisions)
    acted = sum(1 for d in decisions if d.verdict.acts and not d.needs_human)
    escalated = sum(1 for d in decisions if d.needs_human)
    by_model: dict[str, int] = {}
    for d in decisions:
        if d.verdict.acts:
            by_model[d.model_id] = by_model.get(d.model_id, 0) + 1

    models = (
        ", ".join(f"{m} ({n})" for m, n in sorted(by_model.items(), key=lambda kv: -kv[1]))
        or "none"
    )
    return (
        f"Gemma is unavailable, so this is a count rather than an analysis. "
        f"{total} drift events in this window: {acted} actionable, "
        f"{escalated} escalated for human review, {total - acted} suppressed. "
        f"Models with act-worthy drift: {models}. "
        f"Open an individual event for its telemetry."
    )
