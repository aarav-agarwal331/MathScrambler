"""Filesystem locations.

Everything MathScrambler writes outside the repo lives under
``~/Library/Application Support/MathScrambler/`` (plus the uv tool shim).
All getters are functions so tests can redirect them via MATHSCRAMBLER_HOME.
"""

from __future__ import annotations

import os
from pathlib import Path


def app_support_dir() -> Path:
    override = os.environ.get("MATHSCRAMBLER_HOME")
    if override:
        return Path(override).resolve()
    return Path.home() / "Library" / "Application Support" / "MathScrambler"


def runs_dir() -> Path:
    return app_support_dir() / "runs"


def logs_dir() -> Path:
    return app_support_dir() / "logs"


def scratch_dir() -> Path:
    return app_support_dir() / "scratch"


def ollama_log_path() -> Path:
    return logs_dir() / "ollama.log"


def pid_file_path() -> Path:
    return app_support_dir() / "ollama.pid"


def server_lock_path() -> Path:
    return app_support_dir() / "server.lock"


def state_file_path() -> Path:
    """Small JSON blob: resolved binary, capability-probe results, active port."""
    return app_support_dir() / "state.json"


def db_path() -> Path:
    return app_support_dir() / "mathscrambler.db"


def ensure_app_dirs() -> list[Path]:
    """Create the app-support tree; returns the directories (for setup's report)."""
    dirs = [app_support_dir(), runs_dir(), logs_dir(), scratch_dir()]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def repo_root() -> Path:
    """The checkout root (editable install: this file lives in <root>/src/mathscrambler/)."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    # Fallback: walk up from cwd (e.g. running from a source tree without install).
    cur = Path.cwd().resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "mathscrambler").is_dir():
            return parent
    raise RuntimeError(
        "Cannot locate the MathScrambler repo root; reinstall with `uv tool install -e .` from the checkout."
    )


def config_path() -> Path:
    return repo_root() / "config.toml"


def config_example_path() -> Path:
    return repo_root() / "config.example.toml"


def default_model_store() -> Path:
    """The shared Ollama model store.

    Detected, not assumed: honor a globally exported OLLAMA_MODELS if one exists
    (we never *set* it — Section 1.2), else the Ollama default location.
    """
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".ollama" / "models"
