"""Shared fixtures. Unit tests never touch a real Ollama server or the network."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest


@pytest.fixture(autouse=True)
def _env_guard():
    """Fail any test that leaks os.environ mutations (zero-global-side-effects discipline)."""
    def snapshot() -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}

    before = snapshot()
    yield
    after = snapshot()
    added = {k: after[k] for k in after.keys() - before.keys()}
    removed = sorted(before.keys() - after.keys())
    changed = {k: (before[k], after[k]) for k in before.keys() & after.keys() if before[k] != after[k]}
    assert not (added or removed or changed), (
        f"test leaked os.environ mutations: added={added} removed={removed} changed={changed}"
    )


@pytest.fixture()
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all app-support paths into a temp dir."""
    home = tmp_path / "appsupport"
    home.mkdir()
    monkeypatch.setenv("MATHSCRAMBLER_HOME", str(home))
    return home


class _StubHandler(BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, tuple[int, dict]]] = {}

    def do_GET(self) -> None:
        status, body = self.routes.get(self.path, (404, {"error": "not found"}))
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:  # silence
        pass


class HttpStub:
    """Tiny local HTTP server standing in for an Ollama API on an ephemeral port."""

    def __init__(self) -> None:
        handler = type("Handler", (_StubHandler,), {"routes": {}})
        self.handler = handler
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def route(self, path: str, body: dict, status: int = 200) -> None:
        self.handler.routes[path] = (status, body)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def http_stub():
    stub = HttpStub()
    yield stub
    stub.close()
