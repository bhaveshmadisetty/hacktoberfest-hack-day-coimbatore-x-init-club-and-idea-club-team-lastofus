"""Tests for the judgment layer.

Two behaviours matter more than the rest:

- RETRAIN must never survive a pipeline-fault taxonomy. Training on corrupt
  data bakes the bug into the model, and a prompt instruction is not a control.
- Measured numbers must refuse to report themselves when they came from the
  heuristic fallback (AGENTS.md §2 forbids fabricated results).
"""

from __future__ import annotations

from arbiter.evaluator import (
    CONFIDENCE_FLOOR,
    Decision,
    decide,
    gate,
    judge_challenger,
    measure_accuracy,
    measure_suppression,
)
from arbiter.gemma_client import GemmaClient
from arbiter.scenarios import generate_replay
from arbiter.schema import ChallengerReport, DriftEvent, Verdict


class FakeClient(GemmaClient):
    """Returns canned text, so decision logic is tested without the network."""

    def __init__(self, response: str):
        super().__init__(api_key=None, use_cache=False)
        self._response = response
        self.model = "fake"

    @property
    def available(self) -> bool:
        return True

    def generate(self, prompt: str, system: str = "", *, force_live: bool = False) -> str:
        return self._response


def _pipeline_event() -> DriftEvent:
    return next(e for e in generate_replay(20) if e.ground_truth == "upstream_bug")


class TestPipelineFaultGuard:
    def test_retrain_on_pipeline_fault_is_overridden(self):
        """The expensive failure mode, blocked in code rather than in a prompt."""
        client = FakeClient(
            '{"action":"RETRAIN","confidence":0.95,'
            '"reasoning":"Large drift, retrain now.",'
            '"primary_features":[],"taxonomy":"upstream_bug"}'
        )
        decision = decide(_pipeline_event(), client, run_analysis=False)
        assert decision.verdict.action == "INVESTIGATE"
        assert "policy override" in decision.verdict.reasoning
        assert decision.needs_human is True

    def test_schema_break_also_blocks_retrain(self):
        client = FakeClient(
            '{"action":"RETRAIN","confidence":0.9,"reasoning":"go",'
            '"primary_features":[],"taxonomy":"schema_break"}'
        )
        assert decide(_pipeline_event(), client, run_analysis=False).verdict.action == "INVESTIGATE"

    def test_retrain_on_genuine_shift_is_allowed(self):
        """The guard must not block the case retraining actually fixes."""
        event = next(e for e in generate_replay(20) if e.ground_truth == "covariate_shift")
        client = FakeClient(
            '{"action":"RETRAIN","confidence":0.88,'
            '"reasoning":"transaction_amount rose 240 -> 890, real segment.",'
            '"primary_features":["transaction_amount"],"taxonomy":"covariate_shift"}'
        )
        decision = decide(event, client, run_analysis=False)
        assert decision.verdict.action == "RETRAIN"
        assert decision.needs_human is False


class TestConfidenceEscalation:
    def test_low_confidence_escalates_to_human(self):
        client = FakeClient(
            '{"action":"RETRAIN","confidence":0.4,"reasoning":"not sure",'
            '"primary_features":[],"taxonomy":"covariate_shift"}'
        )
        event = next(e for e in generate_replay(20) if e.ground_truth == "covariate_shift")
        assert decide(event, client, run_analysis=False).needs_human is True

    def test_high_confidence_does_not_escalate(self):
        client = FakeClient(
            '{"action":"RETRAIN","confidence":0.95,"reasoning":"clear",'
            '"primary_features":[],"taxonomy":"covariate_shift"}'
        )
        event = next(e for e in generate_replay(20) if e.ground_truth == "covariate_shift")
        assert decide(event, client, run_analysis=False).needs_human is False


class TestHallucinatedFeatureNames:
    def test_features_not_in_the_event_are_dropped(self):
        """The UI renders these as fact, so an invented name must not reach it."""
        event = next(e for e in generate_replay(20) if e.ground_truth == "covariate_shift")
        client = FakeClient(
            '{"action":"BENIGN","confidence":0.8,"reasoning":"fine",'
            '"primary_features":["totally_made_up_feature"],"taxonomy":"seasonal"}'
        )
        decision = decide(event, client, run_analysis=False)
        assert "totally_made_up_feature" not in decision.verdict.primary_features


class TestUnparseableOutput:
    def test_garbage_falls_back_without_crashing(self):
        decision = decide(_pipeline_event(), FakeClient("I cannot help with that."), run_analysis=False)
        assert decision.verdict.source == "heuristic"
        assert decision.verdict.action in ("RETRAIN", "ROLLBACK", "INVESTIGATE", "BENIGN")

    def test_fallback_never_retrains_a_pipeline_fault(self):
        decision = decide(_pipeline_event(), FakeClient("nonsense"), run_analysis=False)
        assert decision.verdict.action != "RETRAIN"


class TestGate:
    def test_suppressed_verdict_does_not_call_train_fn(self):
        """The alert-fatigue win: the retrain that did not happen."""
        calls = []
        client = FakeClient(
            '{"action":"BENIGN","confidence":0.9,"reasoning":"month-end batch",'
            '"primary_features":[],"taxonomy":"seasonal"}'
        )
        event = next(e for e in generate_replay(20) if e.ground_truth == "seasonal")
        decision, result = gate(event, lambda: calls.append(1), client)
        assert calls == [] and result is None and decision.verdict.suppressed

    def test_actionable_verdict_calls_train_fn(self):
        client = FakeClient(
            '{"action":"RETRAIN","confidence":0.92,"reasoning":"real shift",'
            '"primary_features":[],"taxonomy":"covariate_shift"}'
        )
        event = next(e for e in generate_replay(20) if e.ground_truth == "covariate_shift")
        _, result = gate(event, lambda: "new-model", client)
        assert result == "new-model"

    def test_on_suppress_callback_fires(self):
        seen = []
        client = FakeClient(
            '{"action":"BENIGN","confidence":0.9,"reasoning":"x",'
            '"primary_features":[],"taxonomy":"seasonal"}'
        )
        event = next(e for e in generate_replay(20) if e.ground_truth == "seasonal")
        gate(event, None, client, on_suppress=seen.append)
        assert len(seen) == 1


def _decision(action: str, source: str, confidence: float = 0.9, event_id: str = "e1") -> Decision:
    v = Verdict(action=action, confidence=confidence, reasoning="r", source=source)
    v.suppressed = not v.acts
    return Decision(
        event_id=event_id, model_id="m", verdict=v,
        driftguard_would_page=v.suppressed,
        needs_human=confidence < CONFIDENCE_FLOOR or action == "INVESTIGATE",
    )


class TestMeasurementHonesty:
    def test_heuristic_run_is_not_reportable(self):
        """AGENTS.md §2 — fallback numbers measure the fallback, not Gemma."""
        m = measure_suppression([_decision("BENIGN", "heuristic")])
        assert m["reportable"] is False and m["gemma_backed_decisions"] == 0

    def test_gemma_run_is_reportable(self):
        m = measure_suppression([_decision("BENIGN", "gemma:gemma-4-26b-a4b-it")])
        assert m["reportable"] is True

    def test_accuracy_excludes_heuristic_decisions(self):
        """Grading the heuristic against its own source labels is not a result."""
        events = generate_replay(2)
        decisions = [_decision("BENIGN", "heuristic", event_id=e.event_id) for e in events]
        a = measure_accuracy(decisions, events)
        assert a["taxonomy_accuracy_pct"] is None
        assert a["skipped_heuristic"] == 2

    def test_suppression_counts_are_consistent(self):
        decisions = [
            _decision("RETRAIN", "gemma:x", 0.9, "e1"),
            _decision("BENIGN", "gemma:x", 0.9, "e2"),
            _decision("INVESTIGATE", "gemma:x", 0.9, "e3"),
        ]
        m = measure_suppression(decisions)
        assert m["actionable"] == 1
        assert m["actionable"] + m["suppressed"] == len(decisions)

    def test_empty_run_does_not_divide_by_zero(self):
        assert measure_suppression([])["noise_reduction_pct"] == 0.0


class TestChallengerJudgment:
    def test_segment_regression_is_rejected_by_fallback(self):
        """A global gain hiding a segment loss — the UC 6 case."""
        report = ChallengerReport(
            model_id="m", champion_metric=0.912, challenger_metric=0.915,
            champion_segment_metric=0.874, challenger_segment_metric=0.853,
        )
        v = judge_challenger(report, None, FakeClient("not json"))
        assert v.recommendation == "REJECT"

    def test_unmeasured_segment_holds(self):
        report = ChallengerReport(model_id="m", champion_metric=0.9, challenger_metric=0.95)
        assert judge_challenger(report, None, FakeClient("junk")).recommendation == "HOLD"

    def test_invalid_recommendation_defaults_to_hold(self):
        report = ChallengerReport(model_id="m", champion_metric=0.9, challenger_metric=0.91)
        client = FakeClient('{"recommendation":"SHIP IT","confidence":0.9,"reasoning":"yes"}')
        assert judge_challenger(report, None, client).recommendation == "HOLD"
