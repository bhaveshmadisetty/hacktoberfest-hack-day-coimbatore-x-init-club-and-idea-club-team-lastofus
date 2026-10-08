"""Tests for round-robin key rotation and rate-limit failover.

The behaviour that matters: a throttled key must not take the whole dashboard
down with it. One key exhausting its free-tier quota should cost a retry on the
next key, not a degraded narration.
"""

from __future__ import annotations

import time

import pytest

from arbiter.keypool import COOLDOWN_SECONDS, KeyPool, is_rate_limited, reset_pool


@pytest.fixture(autouse=True)
def clean_pool():
    reset_pool()
    yield
    reset_pool()


def pool_of(*keys: str) -> KeyPool:
    return KeyPool([(f"GEMMA_API_KEY_{i + 1}", k) for i, k in enumerate(keys)])


class TestRotation:
    def test_cycles_through_keys_in_order(self):
        pool = pool_of("k1", "k2", "k3")
        got = [pool.acquire().key for _ in range(6)]
        assert got == ["k1", "k2", "k3", "k1", "k2", "k3"]

    def test_single_key_is_reused(self):
        pool = pool_of("only")
        assert [pool.acquire().key for _ in range(3)] == ["only"] * 3

    def test_empty_pool_returns_none(self):
        assert pool_of().acquire() is None

    def test_blank_slots_are_skipped(self):
        """Unfilled GEMMA_API_KEY_n placeholders must not count as keys."""
        pool = KeyPool([("GEMMA_API_KEY", "k1"), ("GEMMA_API_KEY_2", ""), ("GEMMA_API_KEY_3", "   ")])
        assert pool.size == 1

    def test_duplicate_keys_are_collapsed(self):
        """The same key under two aliases is one key, not two."""
        pool = KeyPool([("GEMMA_API_KEY", "same"), ("GOOGLE_API_KEY", "same")])
        assert pool.size == 1


class TestRateLimitFailover:
    def test_throttled_key_is_skipped(self):
        pool = pool_of("k1", "k2")
        first = pool.acquire()
        pool.report_rate_limited(first, "429 quota exceeded")
        # Two acquires: neither should hand back the parked key.
        assert {pool.acquire().key for _ in range(2)} == {"k2"}

    def test_cooldown_is_recorded(self):
        pool = pool_of("k1", "k2")
        slot = pool.acquire()
        pool.report_rate_limited(slot)
        assert slot.available is False
        assert 0 < slot.cooldown_remaining <= COOLDOWN_SECONDS

    def test_all_throttled_returns_the_soonest_to_recover(self):
        """Attempting a likely-429 beats degrading without trying."""
        pool = pool_of("k1", "k2")
        a, b = pool.acquire(), pool.acquire()
        pool.report_rate_limited(a)
        time.sleep(0.01)
        pool.report_rate_limited(b)
        assert pool.acquire() is a

    def test_success_clears_a_cooldown(self):
        pool = pool_of("k1")
        slot = pool.acquire()
        pool.report_rate_limited(slot)
        pool.report_success(slot)
        assert slot.available is True

    def test_non_quota_error_does_not_park_the_key(self):
        """A malformed request is not the key's fault."""
        pool = pool_of("k1", "k2")
        slot = pool.acquire()
        pool.report_error(slot, "ValueError: bad prompt")
        assert slot.available is True


class TestRateLimitDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "ClientError: 429 Too Many Requests",
            "ResourceExhausted: quota exceeded for model",
            "Error: RESOURCE_EXHAUSTED",
            "rate limit reached, retry later",
        ],
    )
    def test_quota_failures_are_detected(self, message):
        assert is_rate_limited(Exception(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "ServerError: 500 INTERNAL",
            "ValueError: prompt too long",
            "ConnectionError: network unreachable",
        ],
    )
    def test_other_failures_are_not(self, message):
        assert is_rate_limited(Exception(message)) is False


class TestStats:
    def test_stats_never_contain_key_material(self):
        """/health is public-ish; a key must not leak through it."""
        pool = pool_of("super-secret-key-value")
        assert "super-secret-key-value" not in str(pool.stats())

    def test_stats_report_pool_health(self):
        pool = pool_of("k1", "k2")
        pool.report_rate_limited(pool.acquire())
        stats = pool.stats()
        assert stats["configured_keys"] == 2
        assert stats["available_now"] == 1
