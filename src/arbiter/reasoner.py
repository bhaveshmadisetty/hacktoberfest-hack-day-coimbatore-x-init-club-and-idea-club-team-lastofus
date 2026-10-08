"""UC 1 + 2 — root-cause narration and drift taxonomy.

Answers "why did this happen?" for one drift event: which features actually
drove it, which of the three causes it is (pipeline fault / population shift /
model decay), and what that means in production.

Every result records whether it came from Gemma or from the heuristic
fallback, so no measured number in the README can silently mix the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gemma_client import GemmaClient, get_client
from .prompts import SYSTEM_REASONER, reasoner_prompt
from .schema import TAXONOMY, DriftEvent

# A feature whose std collapses to this fraction of reference, or whose null
# rate crosses the threshold, is pinned to a default value — the signature of a
# broken upstream join rather than a real population.
_STD_COLLAPSE_RATIO = 0.2
_NULL_RATE_ALERT = 0.05


@dataclass
class Analysis:
    """Root-cause analysis of one DriftEvent."""

    taxonomy: str | None
    primary_features: list[str]
    hypothesis: str
    impact: str = ""
    confidence: float = 0.0
    source: str = "gemma"
    raw: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "taxonomy": self.taxonomy,
            "primary_features": self.primary_features,
            "hypothesis": self.hypothesis,
            "impact": self.impact,
            "confidence": self.confidence,
            "source": self.source,
        }


def _pipeline_signals(event: DriftEvent) -> list[str]:
    """Features showing the pipeline-fault signature, worst drift first."""
    hits = []
    for name, score in sorted(event.feature_scores.items(), key=lambda kv: -kv[1]):
        d = event.delta(name)
        null_jump = d["cur_null_rate"] - d["ref_null_rate"] > _NULL_RATE_ALERT
        std_collapse = (
            d["ref_std"] > 0 and d["cur_std"] / d["ref_std"] < _STD_COLLAPSE_RATIO
        )
        if null_jump or std_collapse:
            hits.append(name)
    return hits


def _heuristic(event: DriftEvent) -> Analysis:
    """Deterministic fallback when Gemma is unavailable or unparseable.

    Deliberately conservative: it reports what the telemetry shows and marks
    itself ``source="heuristic"``. It is not a stand-in for reasoning, and
    results produced this way must not be presented as model output.
    """
    top = event.top_features(3)
    names = [n for n, _ in top]
    broken = _pipeline_signals(event)

    if broken:
        tax = "upstream_bug"
        why = (
            f"{broken[0]} shows the signature of a data pipeline fault "
            f"(null rate {event.delta(broken[0])['cur_null_rate'] * 100:.1f}%, "
            f"variance collapsed toward a constant). Retraining on this window "
            f"would learn the fault."
        )
        conf = 0.55
        names = broken[:3]
    elif event.context.get("live_precision_current") is not None and max(
        (s for _, s in top), default=0.0
    ) < 0.3:
        tax = "concept_drift"
        why = (
            f"Feature distributions are near reference (max drift "
            f"{top[0][1]:.2f} on {top[0][0]}) while live precision fell "
            f"{event.context.get('live_precision_ref')} -> "
            f"{event.context.get('live_precision_current')}. Stable inputs with "
            f"falling performance indicates the input-label relationship moved."
        )
        conf = 0.5
    elif event.global_drift_score < 0.5:
        tax = "seasonal"
        why = (
            f"Moderate global drift ({event.global_drift_score:.2f}) "
            f"concentrated in {top[0][0]} with no nulls and no variance "
            f"collapse. Consistent with a recurring cyclical pattern."
        )
        conf = 0.4
    else:
        tax = "covariate_shift"
        d = event.delta(top[0][0])
        why = (
            f"Drift concentrated in {top[0][0]} "
            f"(ref mean {d['ref_mean']:.1f} -> current {d['cur_mean']:.1f}, "
            f"{d['pct_change']:+.1f}%) with no nulls introduced, so the values "
            f"are internally consistent — a real population shift."
        )
        conf = 0.5

    return Analysis(
        taxonomy=tax,
        primary_features=names,
        hypothesis=why,
        impact="Heuristic fallback: Gemma was unavailable, so no model-generated impact assessment is available.",
        confidence=conf,
        source="heuristic",
    )


def analyze(event: DriftEvent, client: GemmaClient | None = None) -> Analysis:
    """Produce a root-cause analysis for one drift event (UC 1 + 2)."""
    client = client or get_client()
    parsed, raw = client.generate_json(reasoner_prompt(event), SYSTEM_REASONER)

    if not parsed:
        fallback = _heuristic(event)
        fallback.raw = raw
        return fallback

    tax = parsed.get("taxonomy")
    if isinstance(tax, str):
        t = tax.strip().lower().replace(" ", "_").replace("-", "_")
        tax = t if t in TAXONOMY else None
    else:
        tax = None

    feats = parsed.get("primary_features") or []
    if isinstance(feats, str):
        feats = [feats]
    # Keep only names the event actually contains — a hallucinated feature name
    # would be worse than an empty list, since the UI renders these as fact.
    feats = [f for f in (str(x) for x in feats) if f in event.feature_scores]

    hypothesis = str(parsed.get("hypothesis") or "").strip()
    if not hypothesis:
        fallback = _heuristic(event)
        fallback.raw = raw
        return fallback

    try:
        conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0

    return Analysis(
        taxonomy=tax,
        primary_features=feats or [n for n, _ in event.top_features(3)],
        hypothesis=hypothesis,
        impact=str(parsed.get("impact") or "").strip(),
        confidence=conf,
        source=f"gemma:{client.model}",
        raw=raw,
    )


def meets_quality_bar(analysis: Analysis, event: DriftEvent) -> dict[str, bool]:
    """Check a narration against the four quality criteria.

    Generic output ("drift was detected in several features") is the single
    biggest risk to this project, so it is measured rather than assumed. Used
    by the smoke test and reported in the README.
    """
    text = f"{analysis.hypothesis} {analysis.impact}"
    return {
        "names_specific_feature": any(f in text for f in event.feature_scores),
        "cites_magnitude": any(ch.isdigit() for ch in text),
        "assigns_taxonomy": analysis.taxonomy is not None,
        "distinguishes_cause": any(
            kw in text.lower()
            for kw in (
                "pipeline",
                "upstream",
                "null",
                "population",
                "segment",
                "decay",
                "concept",
                "relationship",
                "seasonal",
                "cyclical",
                "join",
                "schema",
            )
        ),
    }
