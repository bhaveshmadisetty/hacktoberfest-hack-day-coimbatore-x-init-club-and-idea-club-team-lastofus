"""Arbiter - AI judgment for production ML, powered by Gemma 4."""

from .env import load_env

# Load .env on import so every entry point (API, CLI, tests) sees the key
# without each one remembering to do it.
load_env()

__version__ = "0.1.0"
