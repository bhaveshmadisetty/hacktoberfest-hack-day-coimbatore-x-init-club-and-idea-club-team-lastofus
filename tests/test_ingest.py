"""Tests for API keys and the SDK-compatible ingest path.

Two properties matter most here:

- A key is never recoverable after creation. The store holds a salted hash,
  so a leaked database does not leak usable credentials.
- The telemetry endpoint stays fast. The SDK calls it from a worker with a 5s
  timeout (tracker.py:_telemetry_worker_loop); blocking it on a ~25s Gemma
  call backs up the queue and drops telemetry, so reasoning must be off the
  request path.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from arbiter import auth, keys
from arbiter.api import app
from arbiter.ingest import _column_stats, _should_reason


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Each test gets its own SQLite file."""
    monkeypatch.setenv("ARBITER_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("ARBITER_CACHE", str(tmp_path / "cache.json"))
    # Drop the cached thread-local connection so the new path is picked up.
    import arbiter.keys as k

    if hasattr(k._LOCAL, "conn"):
        del k._LOCAL.conn
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Bearer header for a fresh account.

    Minting an API key requires a signed-in user, so tests that exercise the
    key lifecycle register one first.
    """
    res = client.post(
        "/auth/register", json={"email": "owner@example.com", "password": "ownerpass123"}
    )
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['token']}"}


FEATURES = [
    "transaction_amount",
    "merchant_category",
    "hour_of_day",
    "card_present",
    "account_age_days",
    "velocity_1h",
]


class TestKeyIssuance:
    def test_created_key_has_prefix_and_is_returned_once(self, client, auth_headers):
        res = client.post("/api/keys", headers=auth_headers, json={"label": "laptop"}).json()
        assert res["key"].startswith("ak_live_")
        assert res["label"] == "laptop"
        assert "DRIFTGUARD_API_KEY" in res["setup"]

    def test_plaintext_is_not_recoverable(self, client, auth_headers):
        key = client.post("/api/keys", headers=auth_headers, json={"label": "x"}).json()["key"]
        listed = client.get("/api/keys", headers=auth_headers).json()["keys"]
        assert all(key not in str(row) for row in listed)
        assert listed[0]["key_hint"].endswith(key[-4:])

    def test_only_a_hash_is_stored(self, client, auth_headers):
        key = client.post("/api/keys", headers=auth_headers, json={"label": "x"}).json()["key"]
        with keys.cursor() as cur:
            cur.execute("SELECT key_hash FROM api_keys")
            stored = cur.fetchone()["key_hash"]
        assert stored != key and len(stored) == 64

    def test_verify_accepts_the_real_key(self, client, auth_headers):
        key = client.post("/api/keys", headers=auth_headers, json={"label": "x"}).json()["key"]
        assert keys.verify_key(key) is not None

    @pytest.mark.parametrize("bad", ["ak_live_wrong", "", None, "garbage"])
    def test_verify_rejects_anything_else(self, client, auth_headers, bad):
        client.post("/api/keys", headers=auth_headers, json={"label": "x"})
        assert keys.verify_key(bad) is None

    def test_revoked_key_stops_working(self, client, auth_headers):
        created = client.post("/api/keys", headers=auth_headers, json={"label": "x"}).json()
        client.delete(f"/api/keys/{created['id']}", headers=auth_headers)
        assert keys.verify_key(created["key"]) is None

    def test_salt_change_invalidates_keys(self, client, auth_headers, monkeypatch):
        """Documents the blast radius of rotating ARBITER_KEY_SALT."""
        key = client.post("/api/keys", headers=auth_headers, json={"label": "x"}).json()["key"]
        monkeypatch.setenv("ARBITER_KEY_SALT", "a-different-salt")
        assert keys.verify_key(key) is None


class TestAuthEnforcement:
    def test_open_until_a_key_exists(self, client):
        """A first-time user can see telemetry before learning the key flow."""
        res = client.post("/models/register", json={"model_id": "m", "features": FEATURES})
        assert res.status_code == 200

    def test_enforced_once_a_key_exists(self, client, auth_headers):
        client.post("/api/keys", headers=auth_headers, json={"label": "x"})
        res = client.post("/models/register", json={"model_id": "m", "features": FEATURES})
        assert res.status_code == 401

    def test_valid_key_is_accepted(self, client, auth_headers):
        key = client.post("/api/keys", headers=auth_headers, json={"label": "x"}).json()["key"]
        res = client.post(
            "/models/register",
            headers={"X-API-Key": key},
            json={"model_id": "m", "features": FEATURES},
        )
        assert res.status_code == 200

    def test_body_api_key_is_not_trusted(self, client, auth_headers):
        """tracker.py puts the key in the body too; only the header counts."""
        client.post("/api/keys", headers=auth_headers, json={"label": "x"})
        res = client.post(
            "/predict/m", json={"model_id": "m", "api_key": "ak_live_fake", "drift_score": 0.1}
        )
        assert res.status_code == 401


class TestSdkProtocol:
    def test_register_stores_ordered_feature_names(self, client):
        """Ordered names are what make a positional drift score narratable."""
        client.post("/models/register", json={"model_id": "m", "features": FEATURES})
        assert keys.feature_names("m") == FEATURES

    def test_register_without_features_warns(self, client):
        res = client.post("/models/register", json={"model_id": "m", "features": []}).json()
        assert res["reasoning_enabled"] is False
        assert "cannot name features" in res["message"]

    def test_register_is_idempotent(self, client):
        for _ in range(3):
            client.post("/models/register", json={"model_id": "m", "features": FEATURES})
        assert len(keys.list_models()) == 1

    def test_telemetry_is_stored(self, client):
        client.post("/models/register", json={"model_id": "m", "features": FEATURES})
        client.post(
            "/predict/m",
            json={"model_id": "m", "features": [1.0] * 6, "prediction": [0], "drift_score": 0.04},
        )
        assert keys.telemetry_count("m") == 1

    def test_telemetry_before_registration_is_kept(self, client):
        """Dropping it would lose the only record of a misconfigured client."""
        client.post("/predict/unknown", json={"model_id": "unknown", "drift_score": 0.1})
        assert keys.telemetry_count("unknown") == 1

    def test_model_lookup_returns_version(self, client):
        client.post(
            "/models/register",
            json={"model_id": "m", "features": FEATURES, "version": "2.1.0"},
        )
        assert client.get("/models/m").json()["version"] == "2.1.0"

    def test_unknown_model_is_404(self, client):
        assert client.get("/models/nope").status_code == 404


class TestReasoningStaysOffTheHotPath:
    def test_low_drift_does_not_queue_reasoning(self, client):
        client.post("/models/register", json={"model_id": "m", "features": FEATURES})
        res = client.post(
            "/predict/m", json={"model_id": "m", "features": [1.0] * 6, "drift_score": 0.02}
        ).json()
        assert res["reasoning_queued"] is False

    def test_no_reasoning_without_feature_names(self, client):
        """Without names a narration could only say "feature 0", so don't spend a call."""
        client.post("/models/register", json={"model_id": "m", "features": []})
        for _ in range(45):
            client.post("/predict/m", json={"model_id": "m", "features": [1.0] * 6, "drift_score": 0.9})
        assert _should_reason("m", 0.9) is False

    def test_no_reasoning_below_sample_floor(self, client):
        """Few samples make reference/current statistics too noisy to narrate."""
        client.post("/models/register", json={"model_id": "m", "features": FEATURES})
        client.post("/predict/m", json={"model_id": "m", "features": [1.0] * 6, "drift_score": 0.9})
        assert _should_reason("m", 0.9) is False

    def test_retrain_is_suppressed_without_a_verdict(self, client):
        """The SDK must not retrain just because its threshold fired."""
        client.post("/models/register", json={"model_id": "m", "features": FEATURES})
        res = client.post("/retrain/m", json={"trigger": "drift_threshold"}).json()
        assert res["status"] == "suppressed"


class TestColumnStats:
    def test_computes_mean_std_and_null_rate(self):
        rows = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        stats = _column_stats(rows, ["a", "b"])
        assert stats["a"]["mean"] == pytest.approx(2.0)
        assert stats["b"]["mean"] == pytest.approx(20.0)
        assert stats["a"]["null_rate"] == 0.0

    def test_counts_nones_as_nulls(self):
        """A null rate jump is the signature of a broken upstream join."""
        stats = _column_stats([[1.0], [None], [3.0], [None]], ["a"])
        assert stats["a"]["null_rate"] == pytest.approx(0.5)

    def test_counts_nan_as_null(self):
        stats = _column_stats([[1.0], [float("nan")]], ["a"])
        assert stats["a"]["null_rate"] == pytest.approx(0.5)

    def test_empty_rows(self):
        assert _column_stats([], ["a"]) == {}

    def test_ragged_rows_do_not_crash(self):
        stats = _column_stats([[1.0, 2.0], [3.0]], ["a", "b"])
        assert "a" in stats
