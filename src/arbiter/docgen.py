"""Gemma-generated incident documentation for one model.

Clicking a model in the dashboard can produce a written report rather than a
dashboard full of numbers: what this model is, every drift incident it has had,
what caused each one, what was decided, and what the operator should do next.

The report is assembled from stored telemetry and verdicts only. Gemma writes
the prose; it is given no freedom to invent an incident, because the incident
list is built here and handed to it. When Gemma is unavailable the report still
renders — as a factual log without narrative, clearly labelled as such, rather
than a blank page.

Output is Markdown, so it can be read in the dashboard, copied into a ticket,
or committed to a repository as a post-mortem.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import keys
from .gemma_client import GemmaClient, get_client

SYSTEM_DOCS = """You are an MLOps engineer writing incident documentation for a production machine-learning model. Your reader is the on-call engineer who inherits this model next week and has no context.

Rules you always follow:
- Write only from the telemetry and verdicts provided. Never invent an incident, a feature name, a number, or a date.
- Cite specific feature names and numeric magnitudes (e.g. "contract_type fell from a mean of 1.68 to 0.068 while nulls rose from 0% to 33%").
- For each incident, state the cause in plain terms and distinguish a data-pipeline failure (nulls appear, variance collapses toward a default) from a genuine population shift (internally consistent values) from model decay (inputs stable, performance falling).
- Be specific about what to do next. "Monitor the situation" is not an action; "check the warehouse enrichment job that populates contract_type" is.
- If the evidence is thin, say so plainly rather than padding.
- Use Markdown with ## headings. No preamble, no sign-off, no "as an AI".
- Keep it tight: an engineer should be able to read the whole thing in two minutes."""


@dataclass
class ModelDoc:
    """A generated report plus the inputs it was built from."""

    model_id: str
    markdown: str
    generated_at: str
    incidents: int
    source: str = "gemma"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "markdown": self.markdown,
            "generated_at": self.generated_at,
            "incidents": self.incidents,
            "source": self.source,
        }


def _fmt(x: float) -> str:
    """Adaptive precision — a ratio printed as 0.3 hides its movement."""
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def build_incident_log(model_id: str, user_id: int | None = None) -> tuple[str, int]:
    """Factual incident log for one model. Returns (log, incident_count).

    This is the evidence block. Gemma narrates it but cannot add to it.
    """
    model = keys.get_model(model_id)
    if not model:
        return f"No model named {model_id} is registered.", 0

    names: list[str] = model.get("features") or []
    telemetry_n = keys.telemetry_count(model_id, user_id=user_id)
    events = keys.list_events(model_id, limit=50, user_id=user_id)

    lines = [
        f"MODEL: {model_id}",
        f"  version            : {model.get('version') or 'unknown'}",
        f"  drift threshold    : {model.get('drift_threshold') or 0.15}",
        f"  reported accuracy  : "
        f"{model.get('accuracy') if model.get('accuracy') is not None else 'not reported'}",
        f"  first registered   : {model.get('registered_at')}",
        f"  last telemetry     : {model.get('last_seen') or 'never'}",
        f"  predictions logged : {telemetry_n}",
        f"  features ({len(names)})      : "
        f"{', '.join(names) if names else 'NONE REGISTERED — narrations cannot name features'}",
    ]

    rows = keys.recent_telemetry(model_id, limit=300, user_id=user_id)
    scores = [r["drift_score"] for r in rows if r.get("drift_score") is not None]
    if scores:
        lines += [
            "",
            "DRIFT SCORE SUMMARY (most recent window):",
            f"  latest {scores[0]:.3f} | mean {statistics.fmean(scores):.3f} "
            f"| max {max(scores):.3f} | min {min(scores):.3f} | n={len(scores)}",
        ]

    if not events:
        lines += [
            "",
            "INCIDENTS: none. Either drift has not crossed the threshold, or not "
            "enough telemetry has accumulated for Arbiter to reason about it.",
        ]
        return "\n".join(lines), 0

    lines += ["", f"INCIDENTS ({len(events)}), newest first:"]
    for i, e in enumerate(events, 1):
        v = e.get("verdict_json") or {}
        a = e.get("analysis_json") or {}
        ev = e.get("event_json") or {}
        ctx = ev.get("context") or {}

        lines += [
            "",
            f"  [{i}] {e['event_id']}  at {e['created_at']}",
            f"      global drift   : {e.get('drift_score')}",
            f"      decision       : {v.get('action')} at {v.get('confidence')} confidence",
            f"      cause          : {v.get('taxonomy') or 'undetermined'}",
            f"      features cited : {', '.join(v.get('primary_features') or []) or 'none'}",
        ]
        if ctx.get("drifted_window_samples"):
            lines.append(
                f"      windows        : {ctx.get('reference_window_samples')} reference "
                f"vs {ctx.get('drifted_window_samples')} drifted samples"
            )
        if v.get("reasoning"):
            lines.append(f"      verdict text   : {v['reasoning']}")
        if a.get("hypothesis"):
            lines.append(f"      root cause     : {a['hypothesis']}")
        if a.get("impact"):
            lines.append(f"      impact         : {a['impact']}")

        # Per-feature movement for the incident, so the report can cite numbers.
        scores_map = ev.get("feature_scores") or {}
        ref = ev.get("reference_stats") or {}
        cur = ev.get("current_stats") or {}
        ranked = sorted(scores_map.items(), key=lambda kv: -kv[1])[:4]
        if ranked:
            lines.append("      feature movement:")
            for name, score in ranked:
                r, c = ref.get(name) or {}, cur.get(name) or {}
                rm, cm = float(r.get("mean", 0.0)), float(c.get("mean", 0.0))
                pct = ((cm - rm) / abs(rm) * 100) if rm else 0.0
                lines.append(
                    f"        {name:<22} drift {score:.2f} | "
                    f"ref mean {_fmt(rm)} std {_fmt(float(r.get('std', 0.0)))} "
                    f"nulls {float(r.get('null_rate', 0.0)) * 100:.1f}% -> "
                    f"cur mean {_fmt(cm)} std {_fmt(float(c.get('std', 0.0)))} "
                    f"nulls {float(c.get('null_rate', 0.0)) * 100:.1f}% ({pct:+.1f}%)"
                )

    return "\n".join(lines), len(events)


def _fallback_markdown(model_id: str, log: str, incidents: int) -> str:
    """A factual report when Gemma is unavailable.

    Deliberately plain: it presents the log and says explicitly that no
    narrative analysis was produced, rather than dressing counts up as prose.
    """
    return f"""# {model_id} — incident log

> Gemma was unavailable when this report was generated, so it contains the
> recorded facts without narrative analysis. Re-generate once reasoning is
> available for the written version.

**Incidents recorded:** {incidents}

## Raw record

```
{log}
```
"""


def generate(
    model_id: str,
    user_id: int | None = None,
    client: GemmaClient | None = None,
) -> ModelDoc:
    """Generate incident documentation for one model."""
    log, incidents = build_incident_log(model_id, user_id)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    client = client or get_client()

    prompt = f"""Write incident documentation for this model, from the record below.

{log}

Produce Markdown with exactly these sections:

## Overview
What this model is, how much telemetry it has produced, and its current state in two or three sentences.

## Incident history
One subsection per incident, newest first, headed `### <event id> — <decision>`. For each: what happened, which features moved and by how much, what Arbiter decided and why, and whether that decision looks right given the evidence. If there are no incidents, say so in one line and omit the rest of this section.

## Root-cause patterns
Whether the incidents share a cause, or whether they are unrelated. Say plainly if there is too little evidence to tell.

## Recommended actions
A short numbered list. Each item names a specific thing to do. If no action is warranted, say that and explain why.

## Monitoring notes
What to watch on this model next, and any gap in the telemetry that limits what can be concluded (for example: features not registered, or a small drifted window).

Markdown only."""

    text = client.generate(prompt, SYSTEM_DOCS).strip()

    if not text:
        return ModelDoc(
            model_id=model_id,
            markdown=_fallback_markdown(model_id, log, incidents),
            generated_at=now,
            incidents=incidents,
            source="fallback",
        )

    # Strip a stray fence — Gemma sometimes wraps whole-document output.
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1 : -1 if lines[-1].strip() == "```" else len(lines)]).strip()

    header = (
        f"*Generated {now} by Arbiter using {client.model}. "
        f"Built from {incidents} recorded incident(s); no facts were added.*\n\n"
    )
    return ModelDoc(
        model_id=model_id,
        markdown=header + text,
        generated_at=now,
        incidents=incidents,
        source=f"gemma:{client.model}",
    )
