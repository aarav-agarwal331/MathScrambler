from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mathscrambler import cli, doctor, ollama_server, setup_flow
from mathscrambler.cli import app
from mathscrambler.doctor import Check, Level

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "mathscramble" in result.output


def test_doctor_green_exits_zero(monkeypatch: pytest.MonkeyPatch):
    checks = [Check("thing", Level.GREEN, "fine")]
    monkeypatch.setattr(cli.doctor_mod, "run_checks", lambda: checks)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "thing" in result.output
    assert "1 green" in result.output


def test_doctor_red_exits_nonzero_and_shows_fix(monkeypatch: pytest.MonkeyPatch):
    checks = [
        Check("broken", Level.RED, "it is broken", fix="run this-one-command"),
        Check("meh", Level.YELLOW, "informational"),
    ]
    monkeypatch.setattr(cli.doctor_mod, "run_checks", lambda: checks)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "this-one-command" in result.output
    assert "1 red" in result.output


def test_server_status_not_running(tmp_home: Path):
    result = runner.invoke(app, ["server", "status"])
    assert result.exit_code == 1
    assert "not running" in result.output


def test_server_stop_with_nothing_running(tmp_home: Path):
    result = runner.invoke(app, ["server", "stop"])
    assert result.exit_code == 0
    assert "nothing to stop" in result.output


def test_server_start_reports_reuse(monkeypatch: pytest.MonkeyPatch, tmp_home: Path):
    info = ollama_server.ServerInfo(
        mode="private", base_url="http://127.0.0.1:21435", port=21435, pid=42, started_by_us=False
    )
    monkeypatch.setattr(cli.ollama_server, "ensure_started", lambda cfg: info)
    monkeypatch.setattr(cli, "load_config", lambda: object())
    result = runner.invoke(app, ["server", "start"])
    assert result.exit_code == 0
    assert "already running" in result.output


def test_setup_delegates(monkeypatch: pytest.MonkeyPatch):
    called = {}
    monkeypatch.setattr(
        cli.setup_flow, "run_setup", lambda console, assume_yes: called.update(yes=assume_yes) or 0
    )
    result = runner.invoke(app, ["setup", "--yes"])
    assert result.exit_code == 0
    assert called == {"yes": True}


# ------------------------------------------------------------------ setup helpers


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("err", request=None, response=None)  # type: ignore[arg-type]

    def json(self):
        return self._payload


def test_registry_size_sums_layers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        setup_flow.httpx,
        "get",
        lambda url, headers=None, timeout=None: _FakeResponse({"layers": [{"size": 10}, {"size": 5}]}),
    )
    assert setup_flow.registry_size("some:tag") == 15


def test_registry_size_handles_404(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        setup_flow.httpx,
        "get",
        lambda url, headers=None, timeout=None: _FakeResponse({"errors": ["x"]}, status=404),
    )
    assert setup_flow.registry_size("glm-5.3-flash") is None


def test_vendor_katex_skips_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    target = tmp_path / "katex"
    target.mkdir()
    (target / "katex.min.css").write_text("/* css */")
    monkeypatch.setattr(setup_flow, "katex_dir", lambda: target)

    def boom(*a, **k):
        raise AssertionError("must not hit the network when already vendored")

    monkeypatch.setattr(setup_flow.httpx, "get", boom)
    from rich.console import Console

    setup_flow.vendor_katex(Console(record=True))


def test_doctor_run_checks_smoke(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    """run_checks on this machine must produce a report without raising, whatever the env."""
    monkeypatch.setattr(doctor, "_launchctl_getenv", lambda name: None)
    monkeypatch.setattr(doctor.ollama_server, "api_version", lambda port, timeout=2.0: None)
    monkeypatch.setattr(doctor.ollama_server, "api_ps", lambda port, timeout=3.0: None)
    monkeypatch.setattr(doctor.ollama_server, "api_tags", lambda port, timeout=5.0: None)
    checks = doctor.run_checks()
    names = [c.name for c in checks]
    assert "ollama binary" in names
    assert "global server (informational)" in names
    assert "memory" in names
    for check in checks:
        if check.level == Level.RED:
            assert check.fix, f"red check {check.name} must carry a fix command"
