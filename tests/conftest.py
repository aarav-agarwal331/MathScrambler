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
    post_scripts: ClassVar[dict[str, list[tuple[int, object]]]] = {}
    post_requests: ClassVar[list[tuple[str, dict]]] = []

    def _send(self, status: int, payload: object) -> None:
        if isinstance(payload, list):  # a list scripts an NDJSON stream response
            data = ("\n".join(json.dumps(item) for item in payload) + "\n").encode()
            ctype = "application/x-ndjson"
        else:
            data = json.dumps(payload).encode()
            ctype = "application/json"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        status, body = self.routes.get(self.path, (404, {"error": "not found"}))
        self._send(status, body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = {}
        self.post_requests.append((self.path, body))
        script = self.post_scripts.get(self.path)
        if not script:
            self._send(404, {"error": "not found"})
        elif len(script) > 1:
            self._send(*script.pop(0))  # consume scripted responses in order
        else:
            self._send(*script[0])  # last response repeats

    def log_message(self, *args: object) -> None:  # silence
        pass


class HttpStub:
    """Tiny local HTTP server standing in for an Ollama API on an ephemeral port."""

    def __init__(self) -> None:
        handler = type(
            "Handler", (_StubHandler,), {"routes": {}, "post_scripts": {}, "post_requests": []}
        )
        self.handler = handler
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def route(self, path: str, body: dict, status: int = 200) -> None:
        self.handler.routes[path] = (status, body)

    def post_route(self, path: str, *responses: object) -> None:
        """Script POST responses for `path`, consumed in order (the last one repeats).
        Each response is a body (dict, or list for NDJSON) or a (status, body) tuple."""
        script: list[tuple[int, object]] = []
        for r in responses:
            script.append(r if isinstance(r, tuple) else (200, r))
        self.handler.post_scripts[path] = script

    @property
    def posts(self) -> list[tuple[str, dict]]:
        return self.handler.post_requests

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def http_stub():
    stub = HttpStub()
    yield stub
    stub.close()
