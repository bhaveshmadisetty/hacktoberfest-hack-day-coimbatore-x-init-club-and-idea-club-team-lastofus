"""Round-robin pool of Gemma API keys with rate-limit failover.

A single free-tier key runs out of quota quickly, and when it does every
narration in the dashboard degrades to the offline fallback at once. The pool
spreads calls across every key provided and, when one is throttled, parks it
for a cooldown and moves to the next.

Keys come from ``GEMMA_API_KEY`` plus ``GEMMA_API_KEY_2..5`` (and the
``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` aliases). Slots left empty are skipped,
so one key is enough to run.

The pool never logs or returns key material — callers get an opaque slot index
and the ``label`` ("key 2 of 5") for display.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# How long a key sits out after being rate limited. Google's free tier
# generally recovers within a minute; a longer park would waste a working key,
# a shorter one would thrash against the same 429.
COOLDOWN_SECONDS = 60.0

# Environment names checked in order. The numbered slots are the rotation pool;
# the aliases let an existing single-key setup keep working untouched.
ENV_NAMES = (
    "GEMMA_API_KEY",
    "GEMMA_API_KEY_2",
    "GEMMA_API_KEY_3",
    "GEMMA_API_KEY_4",
    "GEMMA_API_KEY_5",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


@dataclass
class _Slot:
    """One key and its health. ``key`` never leaves this module."""

    key: str
    env_name: str
    index: int
    calls: int = 0
    rate_limited: int = 0
    errors: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = field(default=None, repr=False)

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.cooldown_until

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self.cooldown_until - time.monotonic())


def is_rate_limited(exc: Exception) -> bool:
    """Whether an exception from the Gemini SDK is a quota/rate-limit failure.

    Matched on text because the SDK raises several exception types for this
    (ClientError, ServerError, ResourceExhausted) and the status code is not
    exposed consistently across versions.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "429",
            "resource_exhausted",
            "resourceexhausted",
            "rate limit",
            "ratelimit",
            "quota",
            "too many requests",
        )
    )


class KeyPool:
    """Thread-safe round-robin pool over the configured keys."""

    def __init__(self, keys: list[tuple[str, str]] | None = None) -> None:
        self._lock = threading.Lock()
        self._cursor = 0
        self._slots: list[_Slot] = []

        pairs = keys if keys is not None else self._from_env()
        seen: set[str] = set()
        for env_name, key in pairs:
            # Strip here, not only in _from_env: an unfilled placeholder in .env
            # can carry trailing whitespace, and a blank slot counted as a key
            # makes the pool look healthier than it is.
            key = (key or "").strip()
            # The same key under two aliases is one key, not two.
            if not key or key in seen:
                continue
            seen.add(key)
            self._slots.append(_Slot(key=key, env_name=env_name, index=len(self._slots)))

    @staticmethod
    def _from_env() -> list[tuple[str, str]]:
        return [(name, (os.getenv(name) or "").strip()) for name in ENV_NAMES]

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def size(self) -> int:
        return len(self._slots)

    def acquire(self) -> _Slot | None:
        """Next usable key, round robin. None when all are cooling down.

        Prefers a key that is not in cooldown. If every key is parked, returns
        the one closest to recovery rather than nothing: a likely-429 attempt
        beats degrading to the offline fallback without trying.
        """
        with self._lock:
            if not self._slots:
                return None

            n = len(self._slots)
            for offset in range(n):
                slot = self._slots[(self._cursor + offset) % n]
                if slot.available:
                    self._cursor = (self._cursor + offset + 1) % n
                    return slot

            return min(self._slots, key=lambda s: s.cooldown_until)

    def report_success(self, slot: _Slot) -> None:
        with self._lock:
            slot.calls += 1
            slot.cooldown_until = 0.0
            slot.last_error = None

    def report_rate_limited(self, slot: _Slot, error: str = "") -> None:
        """Park a throttled key so the next call uses a different one."""
        with self._lock:
            slot.rate_limited += 1
            slot.cooldown_until = time.monotonic() + COOLDOWN_SECONDS
            slot.last_error = error[:200] or "rate limited"

    def report_error(self, slot: _Slot, error: str) -> None:
        """A non-quota failure. Does not park the key — the fault is elsewhere."""
        with self._lock:
            slot.errors += 1
            slot.last_error = error[:200]

    def stats(self) -> dict[str, Any]:
        """Pool health for /health. Contains no key material."""
        with self._lock:
            return {
                "configured_keys": len(self._slots),
                "available_now": sum(1 for s in self._slots if s.available),
                "keys": [
                    {
                        "slot": s.index + 1,
                        "env": s.env_name,
                        "calls": s.calls,
                        "rate_limited": s.rate_limited,
                        "errors": s.errors,
                        "cooling_down": not s.available,
                        "cooldown_remaining_s": round(s.cooldown_remaining, 1),
                        "last_error": s.last_error,
                    }
                    for s in self._slots
                ],
            }


_POOL: KeyPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> KeyPool:
    """Process-wide pool, so rotation state is shared across every caller."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = KeyPool()
        return _POOL


def reset_pool() -> None:
    """Rebuild from the environment. Used by tests."""
    global _POOL
    with _POOL_LOCK:
        _POOL = None
