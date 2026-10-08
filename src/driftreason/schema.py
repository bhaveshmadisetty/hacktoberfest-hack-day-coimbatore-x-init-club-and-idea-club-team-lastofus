"""Data contract for DriftReason.

Everything in this package speaks these two shapes. The reasoner, the
evaluator, the API and the frontend all agree here so the four build tracks
never have to negotiate at merge time.

`DriftEvent.feature_scores` is NAME-keyed. The DriftGuard SDK hands back
integer-indexed scores (``{0: 0.94}``) because the detector only ever sees a
positional feature matrix; a narration built on that is worthless ("feature 0
drifted"). `map_feature_scores` does the translation, and it must be applied
before any event reaches Gemma.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

# Verdict actions. RETRAIN/ROLLBACK act; INVESTIGATE escalates to a human;
# BENIGN suppresses the alert entirely.
ACTIONS = ("RETRAIN", "ROLLBACK", "INVESTIGATE", "BENIGN")

# Drift taxonomy (UC 2). schema_break and upstream_bug are pipeline faults —
# retraining on that data makes things worse, which is the distinction that
# makes the taxonomy worth having.
TAXONOMY = (
    "covariate_shift",
    "label_shift",
    "concept_drift",
    "schema_break",
    "seasonal",
    "upstream_bug",
)


def utc_now() -> str:
    """ISO-8601 UTC timestamp, seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class FeatureStat:
    """Reference vs. current distribution summary for one feature.

    Gemma cannot cite a magnitude it was never given, so every feature carries
    its before/after numbers, not just a drift score.
    """

    mean: float
    std: float
    null_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std, "null_rate": self.null_rate}


@dataclass
class DriftEvent:
    """One drift detection, enriched with everything Gemma needs to reason."""

    model_id: str
    global_drift_score: float
    feature_scores: dict[str, float]  # NAME-keyed — see map_feature_scores
    drift_detected: bool
    reference_stats: dict[str, dict[str, float]]
    current_stats: dict[str, dict[str, float]]
    timestamp: str = field(default_factory=utc_now)
    event_id: str = ""
    # Free-form context the detector cannot know: deploy markers, upstream job
    # status, calendar position. Gemma uses it to separate cause from symptom.
    context: dict[str, Any] = field(default_factory=dict)
    # Hand-labelled cause, scenario fixtures only. Never shown to the model —
    # it exists so accuracy claims in the README trace to a real comparison.
    ground_truth: str | None = None

    def top_features(self, n: int = 3) -> list[tuple[str, float]]:
        """The n most-drifted features, highest score first."""
        return sorted(self.feature_scores.items(), key=lambda kv: -kv[1])[:n]

    def delta(self, feature: str) -> dict[str, float]:
        """Reference to current change for one feature, with percent shift."""
        ref = self.reference_stats.get(feature, {})
        cur = self.current_stats.get(feature, {})
        ref_mean = float(ref.get("mean", 0.0))
        cur_mean = float(cur.get("mean", 0.0))
        pct = ((cur_mean - ref_mean) / abs(ref_mean) * 100.0) if ref_mean else 0.0
        return {
            "ref_mean": ref_mean,
            "cur_mean": cur_mean,
            "ref_std": float(ref.get("std", 0.0)),
            "cur_std": float(cur.get("std", 0.0)),
            "pct_change": round(pct, 1),
            "ref_null_rate": float(ref.get("null_rate", 0.0)),
            "cur_null_rate": float(cur.get("null_rate", 0.0)),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """Gemma's judgment on a DriftEvent (UC 1, 2, 5, 10)."""

    action: str
    confidence: float
    reasoning: str
    primary_features: list[str] = field(default_factory=list)
    taxonomy: str | None = None
    # True when the detector fired but Gemma decided no action was warranted —
    # i.e. the alert DriftGuard would have paged on.
    suppressed: bool = False
    # Which model produced this, or "fallback" when parsing failed. Keeps the
    # README honest about what was actually measured.
    source: str = "gemma"

    def __post_init__(self) -> None:
        self.action = str(self.action).strip().upper()
        if self.action not in ACTIONS:
            self.action = "INVESTIGATE"
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.0
        if self.taxonomy:
            t = str(self.taxonomy).strip().lower().replace(" ", "_").replace("-", "_")
            self.taxonomy = t if t in TAXONOMY else None

    @property
    def acts(self) -> bool:
        """Whether this verdict triggers an automated action."""
        return self.action in ("RETRAIN", "ROLLBACK")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChallengerReport:
    """Champion vs. challenger comparison (UC 6).

    The overall delta is the number that misleads: a challenger can gain
    globally while losing on exactly the segment that drifted.
    """

    model_id: str
    champion_metric: float
    challenger_metric: float
    metric_name: str = "accuracy"
    segment_name: str = "drifted_segment"
    champion_segment_metric: float | None = None
    challenger_segment_metric: float | None = None

    @property
    def overall_delta(self) -> float:
        return round(self.challenger_metric - self.champion_metric, 4)

    @property
    def segment_delta(self) -> float | None:
        if self.champion_segment_metric is None or self.challenger_segment_metric is None:
            return None
        return round(self.challenger_segment_metric - self.champion_segment_metric, 4)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overall_delta"] = self.overall_delta
        d["segment_delta"] = self.segment_delta
        return d


def map_feature_scores(
    feature_scores: Mapping[Any, float],
    feature_names: Sequence[str],
) -> dict[str, float]:
    """Translate DriftGuard's integer-indexed scores into name-keyed scores.

    The SDK returns ``{0: 0.94, 1: 0.71}`` — positional, because the detector
    only sees a feature matrix. Gemma cannot say "transaction_amount drifted"
    from that, so this runs before any event is built.

    Keys that are already names pass through, so an SDK release that starts
    returning names needs no change here. An index outside `feature_names`
    falls back to ``feature_<i>`` rather than dropping the score silently.
    """
    named: dict[str, float] = {}
    for key, score in feature_scores.items():
        if isinstance(key, str) and not key.isdigit():
            named[key] = float(score)
            continue
        idx = int(key)
        name = feature_names[idx] if 0 <= idx < len(feature_names) else f"feature_{idx}"
        named[name] = float(score)
    return named


def build_drift_event(
    status: Mapping[str, Any],
    feature_names: Sequence[str],
    model_id: str,
    reference_stats: Mapping[str, Mapping[str, float]] | None = None,
    current_stats: Mapping[str, Mapping[str, float]] | None = None,
    context: Mapping[str, Any] | None = None,
) -> DriftEvent:
    """Build a DriftEvent from a DriftGuard ``get_status()`` payload.

    This is the single seam between the SDK and DriftReason. The SDK stays an
    unmodified pip dependency; the index-to-name mapping happens here.
    """
    return DriftEvent(
        model_id=model_id,
        global_drift_score=float(status.get("global_drift_score", 0.0)),
        feature_scores=map_feature_scores(status.get("feature_scores", {}), feature_names),
        drift_detected=bool(status.get("drift_detected", False)),
        reference_stats={k: dict(v) for k, v in (reference_stats or {}).items()},
        current_stats={k: dict(v) for k, v in (current_stats or {}).items()},
        context=dict(context or {}),
    )
