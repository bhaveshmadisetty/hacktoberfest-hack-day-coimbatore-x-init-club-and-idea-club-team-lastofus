"""User accounts, password hashing, and session tokens.

Passwords are hashed with PBKDF2-HMAC-SHA256 from the standard library
(600k iterations, per-user random salt). That is a deliberate scope decision:
it avoids adding a native dependency (bcrypt/argon2) for a hackathon build
while still never storing a recoverable password. A production deployment
should move to Argon2id — PBKDF2 is GPU-friendlier than a memory-hard KDF.

Sessions are opaque random tokens stored hashed, same reasoning as API keys:
a leaked database yields nothing usable. Tokens carry an expiry and are
checked on every request.

Ownership is the point of this module. Every API key, model, telemetry row
and drift event belongs to exactly one user, so one account can never read
another's telemetry.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from .keys import cursor

# PBKDF2 cost. High enough to be meaningfully slow to brute force, low enough
# that a login on a laptop stays responsive.
_PBKDF2_ROUNDS = 600_000
_SALT_BYTES = 16
_TOKEN_BYTES = 32

SESSION_DAYS = 14

# Deliberately permissive: this validates shape, not deliverability. Rejecting
# unusual-but-valid addresses is worse than accepting one that bounces.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

MIN_PASSWORD_LENGTH = 8


class AuthError(Exception):
    """Raised for any credential failure.

    One exception type for "no such user" and "wrong password" on purpose:
    distinguishing them tells an attacker which emails are registered.
    """


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> str:
    email = normalise_email(email)
    if not _EMAIL_RE.match(email):
        raise AuthError("Enter a valid email address.")
    return email


def validate_password(password: str) -> str:
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    return password


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (hex_digest, hex_salt). A fresh salt is generated if not given."""
    salt = salt or secrets.token_hex(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ROUNDS
    )
    return digest.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Constant-time comparison, so timing does not leak the hash prefix."""
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---- schema -------------------------------------------------------------


def ensure_auth_schema() -> None:
    """Backfill ownership columns on databases created before accounts existed.

    The tables themselves (and the user_id columns on new databases) are
    created by keys._ensure_schema on first connect. This only handles an
    existing file from an earlier version, where the columns are absent.
    SQLite has no ADD COLUMN IF NOT EXISTS, so a duplicate is caught and
    ignored rather than treated as an error.
    """
    with cursor() as cur:
        for table, column in (
            ("api_keys", "user_id INTEGER"),
            ("models", "user_id INTEGER"),
            ("telemetry", "user_id INTEGER"),
            ("events", "user_id INTEGER"),
        ):
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
            except Exception:  # noqa: BLE001 - column already present
                pass


# ---- accounts -----------------------------------------------------------


def register(email: str, password: str, name: str | None = None) -> dict[str, Any]:
    """Create an account. Raises AuthError if the email is taken."""
    email = validate_email(email)
    validate_password(password)
    pw_hash, salt = hash_password(password)

    with cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            raise AuthError("An account with that email already exists.")
        cur.execute(
            "INSERT INTO users (email, password_hash, password_salt, name, created_at) "
            "VALUES (?,?,?,?,?)",
            (email, pw_hash, salt, (name or "").strip() or None, _iso(_now())),
        )
        user_id = int(cur.lastrowid or 0)

    return {"id": user_id, "email": email, "name": name}


def login(email: str, password: str) -> dict[str, Any]:
    """Verify credentials and open a session. Raises AuthError on failure."""
    email = normalise_email(email)
    with cursor() as cur:
        cur.execute(
            "SELECT id, email, name, password_hash, password_salt FROM users WHERE email = ?",
            (email,),
        )
        row = cur.fetchone()

    # Same error either way — see AuthError.
    if not row or not verify_password(password, row["password_hash"], row["password_salt"]):
        raise AuthError("Incorrect email or password.")

    token = create_session(int(row["id"]))
    with cursor() as cur:
        cur.execute("UPDATE users SET last_login = ? WHERE id = ?", (_iso(_now()), row["id"]))

    return {
        "token": token,
        "user": {"id": int(row["id"]), "email": row["email"], "name": row["name"]},
    }


def get_user(user_id: int) -> dict[str, Any] | None:
    with cursor() as cur:
        cur.execute(
            "SELECT id, email, name, created_at, last_login FROM users WHERE id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def change_email(user_id: int, new_email: str, password: str) -> dict[str, Any]:
    """Change the account email. The current password is required."""
    new_email = validate_email(new_email)

    with cursor() as cur:
        cur.execute(
            "SELECT password_hash, password_salt FROM users WHERE id = ?", (user_id,)
        )
        row = cur.fetchone()
    if not row:
        raise AuthError("Account not found.")
    if not verify_password(password, row["password_hash"], row["password_salt"]):
        raise AuthError("Incorrect password.")

    with cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE email = ? AND id != ?", (new_email, user_id))
        if cur.fetchone():
            raise AuthError("That email is already in use.")
        cur.execute("UPDATE users SET email = ? WHERE id = ?", (new_email, user_id))

    return {"id": user_id, "email": new_email}


def change_password(user_id: int, current_password: str, new_password: str) -> None:
    """Change the password and invalidate every other session."""
    validate_password(new_password)

    with cursor() as cur:
        cur.execute(
            "SELECT password_hash, password_salt FROM users WHERE id = ?", (user_id,)
        )
        row = cur.fetchone()
    if not row:
        raise AuthError("Account not found.")
    if not verify_password(current_password, row["password_hash"], row["password_salt"]):
        raise AuthError("Incorrect current password.")

    pw_hash, salt = hash_password(new_password)
    with cursor() as cur:
        cur.execute(
            "UPDATE users SET password_hash = ?, password_salt = ? WHERE id = ?",
            (pw_hash, salt, user_id),
        )
        # A password change must log out anyone holding an old session.
        cur.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


# ---- sessions -----------------------------------------------------------


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    with cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?,?,?,?)",
            (
                hash_token(token),
                user_id,
                _iso(_now()),
                _iso(_now() + timedelta(days=SESSION_DAYS)),
            ),
        )
    return token


def resolve_session(token: str | None) -> dict[str, Any] | None:
    """Return the user for a session token, or None if invalid/expired."""
    if not token:
        return None
    with cursor() as cur:
        cur.execute(
            """
            SELECT u.id, u.email, u.name, s.expires_at
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ?
            """,
            (hash_token(token),),
        )
        row = cur.fetchone()
    if not row:
        return None

    try:
        if datetime.fromisoformat(row["expires_at"]) < _now():
            end_session(token)
            return None
    except (TypeError, ValueError):
        return None

    return {"id": int(row["id"]), "email": row["email"], "name": row["name"]}


def end_session(token: str) -> None:
    with cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))


def purge_expired_sessions() -> int:
    with cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE expires_at < ?", (_iso(_now()),))
        return cur.rowcount
