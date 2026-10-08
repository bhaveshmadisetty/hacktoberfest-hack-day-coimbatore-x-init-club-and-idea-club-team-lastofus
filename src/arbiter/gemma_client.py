"""Hosted Gemma 4 client with JSON repair and a replayable cache.

Three things go wrong with a hosted model in a live demo, and all three are
handled here rather than at the call sites:

1. **No guaranteed JSON mode.** Structured output is not documented for Gemma
   on the Gemini API, so `repair_json` strips fences, extracts the first
   object, and falls back to an INVESTIGATE verdict instead of raising.
2. **The network.** Every response is written to ``demo_cache.json``, keyed by
   a hash of (model, system, prompt). A cache hit never touches the network,
   so a pre-warmed run replays offline and deterministically.
3. **A missing key.** With no ``GEMMA_API_KEY`` the client stays usable: it
   serves cache hits, and reports ``available == False`` so callers can degrade
   honestly rather than crash.

Gemma 4's weights are openly published; this calls hosted inference of that
open-weight model. It does not run weights locally, and nothing here should be
described as if it did.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .keypool import get_pool, is_rate_limited

# Pinned so a server-side default change cannot silently alter reasoning quality.
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

# Low but non-zero: these are judgment calls, not creative writing, and we want
# reproducible narrations across a demo rehearsal.
DEFAULT_TEMPERATURE = 0.2

_CACHE_LOCK = threading.Lock()


def _cache_path() -> Path:
    return Path(os.getenv("ARBITER_CACHE", "demo_cache.json"))


def repair_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a model response that was asked for JSON.

    Returns None when nothing object-shaped can be recovered, so the caller
    decides what a failure means rather than having a verdict invented here.
    """
    if not text:
        return None

    t = text.strip()

    # ```json ... ``` fences are the most common deviation.
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    try:
        parsed = json.loads(t)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Prose wrapped around an object: take the outermost {...} span.
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            # Trailing commas are the next most common malformation.
            try:
                parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class GemmaClient:
    """Thin wrapper over the Gemini API's Gemma models.

    Not thread-safe for cache writes beyond the module lock; good enough for a
    single API process serving a demo.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        use_cache: bool = True,
    ) -> None:
        self.model = model or os.getenv("GEMMA_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.use_cache = use_cache
        self._api_key = (
            api_key
            or os.getenv("GEMMA_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        # One SDK client per key slot, so rotating does not rebuild clients.
        self._clients: dict[int, Any] = {}
        self._pool = get_pool()
        self._cache: dict[str, str] | None = None
        self.rotations = 0
        # Set on the first failed call so a dead key degrades once, not per event.
        self.last_error: str | None = None
        self.calls_made = 0
        self.cache_hits = 0

    # ---- availability -----------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether a live call is possible. False means cache-only."""
        if self._api_key:
            return self._client_for_key(-1, self._api_key) is not None
        return self._pool.size > 0

    def _client_for_key(self, slot_index: int, key: str) -> Any | None:
        """Build (and cache) an SDK client for one key."""
        cached = self._clients.get(slot_index)
        if cached is not None:
            return cached
        try:
            from google import genai  # lazy: cache-only mode needs no SDK
        except ImportError as exc:
            self.last_error = f"google-genai not installed: {exc}"
            return None
        try:
            client = genai.Client(api_key=key)
        except Exception as exc:  # noqa: BLE001 - init must never kill the process
            self.last_error = f"Gemma client init failed: {exc}"
            return None
        self._clients[slot_index] = client
        return client

    # ---- cache ------------------------------------------------------------

    def _load_cache(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache
        path = _cache_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._cache = raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        if self._cache is None:
            return
        path = _cache_path()
        with _CACHE_LOCK:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8"
                )
            except OSError:
                # A read-only mount must not break reasoning; the cache is an
                # optimisation, not a requirement.
                pass

    def _key(self, prompt: str, system: str) -> str:
        blob = f"{self.model}\x00{self.temperature}\x00{system}\x00{prompt}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    # ---- generation -------------------------------------------------------

    def generate(self, prompt: str, system: str = "", *, force_live: bool = False) -> str:
        """Return raw model text. Serves from cache unless `force_live`.

        Raises nothing: a failure with no cached fallback returns "".
        """
        key = self._key(prompt, system)
        cache = self._load_cache() if self.use_cache else {}

        if self.use_cache and not force_live and key in cache:
            self.cache_hits += 1
            return cache[key]

        config: dict[str, Any] = {"temperature": self.temperature}
        if system:
            config["system_instruction"] = system

        text = ""
        if self._api_key:
            # An explicitly supplied key bypasses the pool (tests, and callers
            # that deliberately pin one credential).
            client = self._client_for_key(-1, self._api_key)
            if client is None:
                return cache.get(key, "")
            try:
                resp = client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                text = (getattr(resp, "text", "") or "").strip()
            except Exception as exc:  # noqa: BLE001 - hosted API, demo must survive
                self.last_error = f"{type(exc).__name__}: {exc}"
                return cache.get(key, "")
        else:
            # Rotate: one attempt per configured key, so a throttled key costs
            # a retry on the next one rather than a failed narration.
            attempts = max(1, self._pool.size)
            for attempt in range(attempts):
                slot = self._pool.acquire()
                if slot is None:
                    return cache.get(key, "")
                client = self._client_for_key(slot.index, slot.key)
                if client is None:
                    self._pool.report_error(slot, self.last_error or "client init failed")
                    continue
                try:
                    resp = client.models.generate_content(
                        model=self.model, contents=prompt, config=config
                    )
                    text = (getattr(resp, "text", "") or "").strip()
                    self._pool.report_success(slot)
                    break
                except Exception as exc:  # noqa: BLE001 - hosted API
                    detail = f"{type(exc).__name__}: {exc}"
                    self.last_error = detail
                    if is_rate_limited(exc):
                        self._pool.report_rate_limited(slot, detail)
                        if attempt < attempts - 1:
                            self.rotations += 1
                            continue
                    else:
                        self._pool.report_error(slot, detail)
                    return cache.get(key, "")

        if not text:
            return cache.get(key, "")

        self.calls_made += 1
        if text and self.use_cache:
            cache[key] = text
            self._cache = cache
            self._save_cache()
        return text

    def generate_json(
        self, prompt: str, system: str = "", *, force_live: bool = False
    ) -> tuple[dict[str, Any] | None, str]:
        """Generate and parse JSON. Returns (parsed_or_None, raw_text)."""
        raw = self.generate(prompt, system, force_live=force_live)
        return repair_json(raw), raw

    def stats(self) -> dict[str, Any]:
        """Call accounting, so README numbers can cite live vs. replayed."""
        return {
            "model": self.model,
            "available": self.available,
            "live_calls": self.calls_made,
            "cache_hits": self.cache_hits,
            "cached_responses": len(self._load_cache()),
            "last_error": self.last_error,
            "key_rotations": self.rotations,
            "key_pool": self._pool.stats(),
        }


_DEFAULT: GemmaClient | None = None


def get_client() -> GemmaClient:
    """Process-wide client, so cache and call counts are shared."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GemmaClient()
    return _DEFAULT
