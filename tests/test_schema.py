"""Tests for the data contract.

The index-to-name mapping is the highest-value code in the build: without it
every narration says "feature 0 drifted" and the whole product thesis fails.
It is tested first and hardest.
"""

from __future__ import annotations

import pytest

from arbiter.schema import (
    DriftEvent,
    Verdict,
    ChallengerReport,
    build_drift_event,
    map_feature_scores,
)


class TestMapFeatureScores:
    def test_maps_integer_indices_to_names(self):
        """The SDK returns positional keys; Gemma needs names."""
        assert map_feature_scores(
            {0: 0.94, 1: 0.71}, ["transaction_amount", "merchant_category"]
        ) == {"transaction_amount": 0.94, "merchant_category": 0.71}

    def test_passes_through_names_unchanged(self):
        """An SDK release returning names must not need a code change here."""
        assert map_feature_scores({"amount": 0.5}, ["amount"]) == {"amount": 0.5}

    def test_digit_strings_are_treated_as_indices(self):
        """JSON round-trips turn int keys into strings."""
        assert map_feature_scores({"0": 0.3}, ["amount"]) == {"amount": 0.3}

    def test_out_of_range_index_is_kept_not_dropped(self):
        """A dropped score is a silently missing feature — worse than a label."""
        assert map_feature_scores({0: 0.5, 9: 0.1}, ["a"]) == {"a": 0.5, "feature_9": 0.1}

    def test_empty_input(self):
        assert map_feature_scores({}, ["a"]) == {}


class TestVerdict:
    @pytest.mark.parametrize("bad", ["maybe", "", "retrain now", "unknown"])
    def test_unrecognised_action_becomes_investigate(self, bad):
        """An unparseable action must escalate, never act."""
        assert Verdict(action=bad, confidence=0.9, reasoning="x").action == "INVESTIGATE"

    def test_lowercase_action_is_accepted(self):
        assert Verdict(action="retrain", confidence=0.5, reasoning="x").action == "RETRAIN"

    @pytest.mark.parametrize(
        "raw,expected", [(5.0, 1.0), (-2.0, 0.0), ("0.7", 0.7), (None, 0.0), ("abc", 0.0)]
    )
    def test_confidence_is_clamped_and_coerced(self, raw, expected):
        assert Verdict(action="BENIGN", confidence=raw, reasoning="x").confidence == expected

    def test_taxonomy_is_normalised(self):
        v = Verdict(action="BENIGN", confidence=0.5, reasoning="x", taxonomy="Covariate-Shift")
        assert v.taxonomy == "covariate_shift"

    def test_invalid_taxonomy_becomes_none(self):
        v = Verdict(action="BENIGN", confidence=0.5, reasoning="x", taxonomy="vibes")
        assert v.taxonomy is None

    def test_acts_only_for_retrain_and_rollback(self):
        acting = {"RETRAIN", "ROLLBACK"}
        for action in ("RETRAIN", "ROLLBACK", "INVESTIGATE", "BENIGN"):
            v = Verdict(action=action, confidence=0.9, reasoning="x")
            assert v.acts is (action in acting)


class TestDriftEvent:
    def test_delta_computes_percent_change(self):
        e = DriftEvent(
            "m", 0.8, {"amount": 0.9}, True,
            {"amount": {"mean": 240.0, "std": 50.0}},
            {"amount": {"mean": 890.0, "std": 120.0}},
        )
        d = e.delta("amount")
        assert d["pct_change"] == pytest.approx(270.8, abs=0.1)

    def test_delta_handles_zero_reference_mean(self):
        """Division by zero must not crash a narration."""
        e = DriftEvent("m", 0.5, {"f": 0.5}, True, {"f": {"mean": 0.0, "std": 1.0}}, {"f": {"mean": 5.0, "std": 1.0}})
        assert e.delta("f")["pct_change"] == 0.0

    def test_delta_of_unknown_feature_returns_zeros(self):
        e = DriftEvent("m", 0.5, {}, True, {}, {})
        assert e.delta("ghost")["ref_mean"] == 0.0

    def test_top_features_ranks_by_score(self):
        e = DriftEvent("m", 0.5, {"a": 0.1, "b": 0.9, "c": 0.5}, True, {}, {})
        assert [n for n, _ in e.top_features(2)] == ["b", "c"]


class TestBuildDriftEvent:
    def test_maps_sdk_status_payload(self):
        """The single seam between the unmodified SDK and Arbiter."""
        event = build_drift_event(
            status={"global_drift_score": 0.87, "feature_scores": {0: 0.94, 1: 0.2}, "drift_detected": True},
            feature_names=["transaction_amount", "hour_of_day"],
            model_id="fraud-v2",
        )
        assert event.feature_scores == {"transaction_amount": 0.94, "hour_of_day": 0.2}
        assert event.drift_detected is True

    def test_tolerates_missing_status_keys(self):
        event = build_drift_event({}, ["a"], "m")
        assert event.global_drift_score == 0.0 and event.feature_scores == {}


class TestChallengerReport:
    def test_segment_delta_can_be_negative_while_overall_is_positive(self):
        """The case UC 6 exists for: a global gain hiding a segment loss."""
        r = ChallengerReport(
            model_id="m", champion_metric=0.912, challenger_metric=0.915,
            champion_segment_metric=0.874, challenger_segment_metric=0.853,
        )
        assert r.overall_delta > 0 and r.segment_delta < 0

    def test_segment_delta_is_none_when_unmeasured(self):
        r = ChallengerReport(model_id="m", champion_metric=0.9, challenger_metric=0.91)
        assert r.segment_delta is None
