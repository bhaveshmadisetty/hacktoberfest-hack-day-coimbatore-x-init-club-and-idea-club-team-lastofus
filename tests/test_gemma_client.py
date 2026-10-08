"""Tests for JSON repair, caching, and degraded operation.

Gemma has no guaranteed JSON mode on the Gemini API, so these malformations
are expected traffic rather than edge cases. A demo must survive all of them.
"""

from __future__ import annotations

import json

import pytest

from arbiter.gemma_client import GemmaClient, repair_json
from arbiter.keypool import reset_pool


class TestRepairJson:
    def test_plain_object(self):
        assert repair_json('{"action":"RETRAIN"}') == {"action": "RETRAIN"}

    def test_strips_json_fence(self):
        assert repair_json('```json\n{"action":"BENIGN"}\n```') == {"action": "BENIGN"}

    def test_strips_bare_fence(self):
        assert repair_json('```\n{"a":1}\n```') == {"a": 1}

    def test_extracts_object_from_surrounding_prose(self):
        raw = 'Here is my analysis: {"action":"ROLLBACK","confidence":0.7} hope that helps'
        assert repair_json(raw) == {"action": "ROLLBACK", "confidence": 0.7}

    def test_repairs_trailing_commas(self):
        assert repair_json('{"features":["a","b",],}') == {"features": ["a", "b"]}

    def test_nested_object_survives_outermost_span_match(self):
        raw = 'text {"a":{"b":2},"c":[1,2]} more text'
        assert repair_json(raw) == {"a": {"b": 2}, "c": [1, 2]}

    @pytest.mark.parametrize("raw", ["", None, "I cannot answer that.", "   "])
    def test_unrecoverable_returns_none(self, raw):
        """None means 'caller decides' — a verdict is never invented here."""
        assert repair_json(raw) is None

    def test_top_level_array_is_rejected(self):
        """Callers expect an object; an array would KeyError downstream."""
        assert repair_json("[1,2,3]") is None


@pytest.fixture
def no_keys(monkeypatch):
    """Remove every key AND rebuild the pool.

    The pool reads the environment once at construction, so clearing the vars
    without resetting it leaves the real key in place and the "degraded" path
    silently makes live calls.
    """
    for var in (
        "GEMMA_API_KEY",
        "GEMMA_API_KEY_2",
        "GEMMA_API_KEY_3",
        "GEMMA_API_KEY_4",
        "GEMMA_API_KEY_5",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_pool()
    yield
    reset_pool()


class TestGemmaClientDegraded:
    def test_no_key_means_unavailable(self, no_keys):
        assert GemmaClient(use_cache=False).available is False

    def test_generate_returns_empty_without_key_or_cache(self, no_keys):
        """Degrades to "" rather than raising — the demo must not crash."""
        assert GemmaClient(use_cache=False).generate("prompt") == ""

    def test_generate_json_returns_none_without_key(self, no_keys):
        parsed, raw = GemmaClient(use_cache=False).generate_json("p")
        assert parsed is None and raw == ""


class TestCacheReplay:
    def test_cache_hit_serves_without_a_key(self, tmp_path, monkeypatch, no_keys):
        """The demo-survival path: a pre-warmed cache replays offline."""
        cache_file = tmp_path / "demo_cache.json"
        monkeypatch.setenv("ARBITER_CACHE", str(cache_file))

        warm = GemmaClient()
        key = warm._key("my prompt", "my system")
        cache_file.write_text(json.dumps({key: '{"action":"BENIGN"}'}), encoding="utf-8")

        cold = GemmaClient()
        parsed, _ = cold.generate_json("my prompt", "my system")
        assert parsed == {"action": "BENIGN"}
        assert cold.cache_hits == 1

    def test_cache_key_varies_with_prompt_system_and_model(self, monkeypatch):
        monkeypatch.setenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
        c = GemmaClient(use_cache=False)
        base = c._key("p", "s")
        assert base != c._key("p2", "s")
        assert base != c._key("p", "s2")
        other = GemmaClient(model="gemma-4-31b-it", use_cache=False)
        assert base != other._key("p", "s")

    def test_corrupt_cache_file_is_ignored(self, tmp_path, monkeypatch):
        """A truncated cache must not take down the API."""
        cache_file = tmp_path / "demo_cache.json"
        cache_file.write_text("{not json", encoding="utf-8")
        monkeypatch.setenv("ARBITER_CACHE", str(cache_file))
        assert GemmaClient()._load_cache() == {}
