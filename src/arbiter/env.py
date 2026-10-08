"""Minimal .env loader.

Avoids a python-dotenv dependency for one small job (AGENTS.md §2: no
unnecessary dependencies). Real environment variables always win, so a
container passing GEMMA_API_KEY directly is never overridden by a stray file.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> list[str]:
    """Load KEY=VALUE pairs from `path`. Returns the names that were set.

    Values are not logged — the caller gets names only, so a key cannot leak
    into stdout or a traceback.
    """
    p = Path(path)
    if not p.is_file():
        return []

    loaded: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
