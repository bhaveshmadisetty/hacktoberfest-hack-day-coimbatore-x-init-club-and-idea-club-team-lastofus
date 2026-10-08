"""Arbiter API keys and the telemetry store.

Arbiter issues its own keys. A user generates one in the dashboard, puts it in
their own code as ``DRIFTGUARD_API_KEY``, points ``DRIFTGUARD_API_URL`` at
Arbiter, and their telemetry starts arriving here — no SDK fork, no signup with
anyone else, nothing to ask a third party for.

Keys are stored as salted SHA-256 hashes, so the database never holds a usable
credential. The plaintext key is returned exactly once, at creation, and cannot
be recovered afterwards — the same contract Vercel and Stripe use, and the
reason the dashboard warns you to copy it immediately.

SQLite because the whole point is that a team can run this themselves. One
file, no server, and WAL mode so the API can read while telemetry writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# ak_live_ prefix so a leaked key is recognisable in logs and git history.
KEY_PREFIX = "ak_live_"
KEY_BYTES = 24

_LOCAL = threading.local()
_INIT_LOCK = threading.Lock()
_INITIALISED: set[str] = set()


def db_path() -> str:
    return os.getenv("ARBITER_DB", "./arbiter.db")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def hash_key(key: str) -> str:
    """Hash a key for storage. Never store or log the plaintext."""
    salt = os.getenv("ARBITER_KEY_SALT", "arbiter-dev-salt")
    return hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    """One connection per thread — sqlite3 objects are not thread-safe."""
    conn = getattr(_LOCAL, "conn", None)
    path = db_path()
    if conn is not None and getattr(_LOCAL, "path", None) == path:
        return conn

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL lets the dashboard read while the SDK writes telemetry.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _LOCAL.conn = conn
    _LOCAL.path = path
    _ensure_schema(conn, path)
    return conn


def _ensure_schema(conn: sqlite3.Connection, path: str) -> None:
    with _INIT_LOCK:
        if path in _INITIALISED:
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash    TEXT    NOT NULL UNIQUE,
                key_hint    TEXT    NOT NULL,   -- ak_live_…abcd, safe to display
                label       TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                last_used   TEXT,
                revoked     INTEGER NOT NULL DEFAULT 0,
                user_id     INTEGER
            );

            CREATE TABLE IF NOT EXISTS models (
                model_id        TEXT PRIMARY KEY,
                key_id          INTEGER,
                features        TEXT NOT NULL DEFAULT '[]',  -- JSON array, ordered
                drift_threshold REAL,
                version         TEXT,
                accuracy        REAL,
                registered_at   TEXT NOT NULL,
                last_seen       TEXT,
                user_id         INTEGER,
                FOREIGN KEY (key_id) REFERENCES api_keys(id)
            );

            -- Raw per-prediction telemetry from the SDK.
            CREATE TABLE IF NOT EXISTS telemetry (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id    TEXT NOT NULL,
                received_at TEXT NOT NULL,
                drift_score REAL,
                features    TEXT,   -- JSON array
                prediction  TEXT,   -- JSON
                user_id     INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_telemetry_model
                ON telemetry(model_id, id DESC);

            -- One row per drift event Arbiter reasoned about.
            CREATE TABLE IF NOT EXISTS events (
                event_id      TEXT PRIMARY KEY,
                model_id      TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                drift_score   REAL,
                event_json    TEXT NOT NULL,
                verdict_json  TEXT,
                analysis_json TEXT,
                user_id       INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_events_model
                ON events(model_id, created_at DESC);

            -- Accounts. Defined here rather than in auth.py so that EVERY
            -- database gets them on first connect: tests point ARBITER_DB at a
            -- fresh file per test, and a one-shot migration at import time
            -- would leave those databases without the tables.
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                password_salt TEXT    NOT NULL,
                name          TEXT,
                created_at    TEXT    NOT NULL,
                last_login    TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                created_at TEXT    NOT NULL,
                expires_at TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            """
        )
        conn.commit()
        _INITIALISED.add(path)


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    conn = _connect()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ---- keys ----------------------------------------------------------------


def create_key(label: str = "default", user_id: int | None = None) -> dict[str, Any]:
    """Mint a new API key. The plaintext is returned once and never stored."""
    key = KEY_PREFIX + secrets.token_urlsafe(KEY_BYTES)
    hint = f"{KEY_PREFIX}…{key[-4:]}"
    with cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (key_hash, key_hint, label, created_at, user_id) "
            "VALUES (?,?,?,?,?)",
            (hash_key(key), hint, label.strip() or "default", _now(), user_id),
        )
        key_id = cur.lastrowid
    return {"id": key_id, "key": key, "key_hint": hint, "label": label, "created_at": _now()}


def verify_key(key: str | None) -> dict[str, Any] | None:
    """Return the key record if valid and not revoked, else None."""
    if not key:
        return None
    with cursor() as cur:
        cur.execute(
            "SELECT id, key_hint, label, revoked, user_id FROM api_keys WHERE key_hash = ?",
            (hash_key(key),),
        )
        row = cur.fetchone()
        if row is None or row["revoked"]:
            return None
        cur.execute("UPDATE api_keys SET last_used = ? WHERE id = ?", (_now(), row["id"]))
        return dict(row)


def list_keys(user_id: int | None = None) -> list[dict[str, Any]]:
    """Keys for one user, hints only — plaintext is unrecoverable by design."""
    sql = (
        "SELECT id, key_hint, label, created_at, last_used, revoked "
        "FROM api_keys"
    )
    params: list[Any] = []
    if user_id is not None:
        sql += " WHERE user_id = ?"
        params.append(user_id)
    sql += " ORDER BY id DESC"
    with cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def revoke_key(key_id: int, user_id: int | None = None) -> bool:
    """Revoke a key. Scoped by user so one account cannot revoke another's."""
    with cursor() as cur:
        if user_id is None:
            cur.execute("UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,))
        else:
            cur.execute(
                "UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_id = ?",
                (key_id, user_id),
            )
        return cur.rowcount > 0


def has_any_key() -> bool:
    with cursor() as cur:
        cur.execute("SELECT 1 FROM api_keys WHERE revoked = 0 LIMIT 1")
        return cur.fetchone() is not None


# ---- models --------------------------------------------------------------


def register_model(
    model_id: str,
    features: list[str],
    key_id: int | None = None,
    drift_threshold: float | None = None,
    version: str | None = None,
    accuracy: float | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    """Record a model and its ORDERED feature names.

    The feature order is the whole reason this table exists: the SDK's drift
    scores arrive positionally, and these names are what turn index 0 into
    "transaction_amount" so a narration can be specific.
    """
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO models (model_id, key_id, features, drift_threshold,
                                version, accuracy, registered_at, last_seen, user_id)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(model_id) DO UPDATE SET
                features        = excluded.features,
                drift_threshold = excluded.drift_threshold,
                version         = excluded.version,
                accuracy        = excluded.accuracy,
                last_seen       = excluded.last_seen,
                -- Never orphan a model: keep the existing owner when a
                -- re-registration arrives without one.
                user_id         = COALESCE(excluded.user_id, models.user_id)
            """,
            (
                model_id,
                key_id,
                json.dumps(list(features)),
                drift_threshold,
                version,
                accuracy,
                _now(),
                _now(),
                user_id,
            ),
        )
    return get_model(model_id) or {}


def get_model(model_id: str) -> dict[str, Any] | None:
    with cursor() as cur:
        cur.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["features"] = json.loads(d.get("features") or "[]")
        return d


def list_models(user_id: int | None = None) -> list[dict[str, Any]]:
    with cursor() as cur:
        if user_id is None:
            cur.execute("SELECT * FROM models ORDER BY last_seen DESC")
        else:
            cur.execute(
                "SELECT * FROM models WHERE user_id = ? ORDER BY last_seen DESC",
                (user_id,),
            )
        out = []
        for row in cur.fetchall():
            d = dict(row)
            d["features"] = json.loads(d.get("features") or "[]")
            out.append(d)
        return out


def feature_names(model_id: str) -> list[str]:
    """Ordered feature names for a model, or [] if never registered."""
    model = get_model(model_id)
    return model["features"] if model else []


# ---- telemetry -----------------------------------------------------------


def record_telemetry(
    model_id: str,
    features: Any = None,
    prediction: Any = None,
    drift_score: float | None = None,
    user_id: int | None = None,
) -> int:
    with cursor() as cur:
        cur.execute(
            "INSERT INTO telemetry (model_id, received_at, drift_score, features, "
            "prediction, user_id) VALUES (?,?,?,?,?,?)",
            (
                model_id,
                _now(),
                drift_score,
                json.dumps(features) if features is not None else None,
                json.dumps(prediction) if prediction is not None else None,
                user_id,
            ),
        )
        cur.execute("UPDATE models SET last_seen = ? WHERE model_id = ?", (_now(), model_id))
        return int(cur.lastrowid or 0)


def recent_telemetry(
    model_id: str | None = None, limit: int = 100, user_id: int | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM telemetry"
    params: list[Any] = []
    where = []
    if model_id:
        where.append("model_id = ?")
        params.append(model_id)
    if user_id is not None:
        where.append("user_id = ?")
        params.append(user_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with cursor() as cur:
        cur.execute(sql, params)
        out = []
        for row in cur.fetchall():
            d = dict(row)
            for field in ("features", "prediction"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except json.JSONDecodeError:
                        pass
            out.append(d)
        return out


def telemetry_count(model_id: str | None = None, user_id: int | None = None) -> int:
    with cursor() as cur:
        if model_id and user_id is not None:
            cur.execute(
                "SELECT COUNT(*) AS n FROM telemetry WHERE model_id = ? AND user_id = ?",
                (model_id, user_id),
            )
        elif model_id:
            cur.execute("SELECT COUNT(*) AS n FROM telemetry WHERE model_id = ?", (model_id,))
        elif user_id is not None:
            cur.execute("SELECT COUNT(*) AS n FROM telemetry WHERE user_id = ?", (user_id,))
        else:
            cur.execute("SELECT COUNT(*) AS n FROM telemetry")
        return int(cur.fetchone()["n"])


# ---- events --------------------------------------------------------------


def save_event(
    event_id: str,
    model_id: str,
    drift_score: float,
    event_json: dict[str, Any],
    verdict_json: dict[str, Any] | None = None,
    analysis_json: dict[str, Any] | None = None,
    user_id: int | None = None,
) -> None:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO events (event_id, model_id, created_at, drift_score,
                                event_json, verdict_json, analysis_json, user_id)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(event_id) DO UPDATE SET
                verdict_json  = excluded.verdict_json,
                analysis_json = excluded.analysis_json
            """,
            (
                event_id,
                model_id,
                _now(),
                drift_score,
                json.dumps(event_json),
                json.dumps(verdict_json) if verdict_json else None,
                json.dumps(analysis_json) if analysis_json else None,
                user_id,
            ),
        )


def list_events(
    model_id: str | None = None, limit: int = 100, user_id: int | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM events"
    params: list[Any] = []
    where = []
    if model_id:
        where.append("model_id = ?")
        params.append(model_id)
    if user_id is not None:
        where.append("user_id = ?")
        params.append(user_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    params.append(limit)

    with cursor() as cur:
        cur.execute(sql, params)
        out = []
        for row in cur.fetchall():
            d = dict(row)
            for field in ("event_json", "verdict_json", "analysis_json"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except json.JSONDecodeError:
                        d[field] = None
            out.append(d)
        return out


def reset_db() -> None:
    """Drop everything, accounts included. Used by tests."""
    with cursor() as cur:
        for table in (
            "telemetry",
            "events",
            "models",
            "api_keys",
            "sessions",
            "users",
        ):
            # sessions/users do not exist until ensure_auth_schema() has run.
            try:
                cur.execute(f"DELETE FROM {table}")
            except Exception:  # noqa: BLE001 - table not created yet
                pass
