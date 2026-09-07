from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mathscrambler import cli, doctor, models_cmd, ollama_server, paths, setup_flow
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


def test_server_stop_already_gone_exits_zero(monkeypatch: pytest.MonkeyPatch):
    """Desired end state (no server) holds -> exit 0, even though nothing was signaled."""
    gone = ollama_server.StopResult(ok=True, stopped=False, reason="server already gone; cleaned")
    monkeypatch.setattr(cli.ollama_server, "stop", lambda: gone)
    result = runner.invoke(app, ["server", "stop"])
    assert result.exit_code == 0


def test_server_stop_unstoppable_exits_nonzero(monkeypatch: pytest.MonkeyPatch):
    stuck = ollama_server.StopResult(ok=False, stopped=False, reason="could not stop pid 1", pid=1)
    monkeypatch.setattr(cli.ollama_server, "stop", lambda: stuck)
    result = runner.invoke(app, ["server", "stop"])
    assert result.exit_code == 1


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
    seen: dict = {}

    def fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        return _FakeResponse({"layers": [{"size": 10}, {"size": 5}]})

    monkeypatch.setattr(setup_flow.httpx, "get", fake_get)
    assert setup_flow.registry_size("some:tag") == 15
    assert "/v2/library/some/manifests/tag" in seen["url"]
    assert setup_flow.registry_size("user/model:tag") == 15
    assert "/v2/user/model/manifests/tag" in seen["url"], "namespaced tags must not get library/ prefixed"


def test_pull_approval_gate_never_disarms_on_unknown_size():
    """SPEC 1.4: >30 GB needs an OK — and unknown size could be >30 GB, so it asks too."""
    assert setup_flow.needs_pull_approval(65 * 1024**3, assume_yes=False)
    assert setup_flow.needs_pull_approval(None, assume_yes=False)
    assert not setup_flow.needs_pull_approval(10 * 1024**3, assume_yes=False)
    assert not setup_flow.needs_pull_approval(None, assume_yes=True)
    assert not setup_flow.needs_pull_approval(65 * 1024**3, assume_yes=True)


def test_registry_size_handles_404(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        setup_flow.httpx,
        "get",
        lambda url, headers=None, timeout=None: _FakeResponse({"errors": ["x"]}, status=404),
    )
    assert setup_flow.registry_size("glm-5.3-flash") is None


def test_vendor_katex_skips_only_when_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    target = tmp_path / "katex"
    target.mkdir()
    monkeypatch.setattr(setup_flow, "katex_dir", lambda: target)
    network_hits: list = []

    def fake_get(*a, **k):
        network_hits.append(a)
        raise setup_flow.httpx.ConnectError("offline")

    monkeypatch.setattr(setup_flow.httpx, "get", fake_get)
    from rich.console import Console

    # Partial previous attempt (css only): must RE-vendor, not skip.
    (target / "katex.min.css").write_text("/* css */")
    setup_flow.vendor_katex(Console(record=True))
    assert network_hits, "a partial vendor must be re-attempted"

    # Complete install: must skip without touching the network.
    for name in ("katex.min.js", "auto-render.min.js"):
        (target / name).write_text("js")
    (target / "fonts").mkdir()
    network_hits.clear()
    setup_flow.vendor_katex(Console(record=True))
    assert not network_hits, "a complete vendor must not hit the network"


# ------------------------------------------------------------------ models command


@pytest.fixture()
def example_cfg(monkeypatch: pytest.MonkeyPatch):
    from mathscrambler.config import load_config

    cfg = load_config(paths.config_example_path())
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    return cfg


def test_models_lists_pulled_and_missing(monkeypatch: pytest.MonkeyPatch, example_cfg, tmp_home: Path):
    monkeypatch.setattr(
        models_cmd, "_installed_tags", lambda cfg: ({"gpt-oss:120b": 65_000_000_000}, "private:11435")
    )
    monkeypatch.setattr(models_cmd, "_residents", lambda cfg: {"gpt-oss:120b": "private"})
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "reasoner" in result.output
    assert "not pulled" in result.output  # vision tag absent from installed
    assert "65.0 GB" in result.output.replace("\n", "")
    assert "private" in result.output


def test_models_list_without_any_server(monkeypatch: pytest.MonkeyPatch, example_cfg, tmp_home: Path):
    monkeypatch.setattr(models_cmd, "_installed_tags", lambda cfg: None)
    monkeypatch.setattr(models_cmd, "_residents", lambda cfg: {})
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "no reachable server" in result.output


def test_models_pull_fetches_missing_only(monkeypatch: pytest.MonkeyPatch, example_cfg, tmp_home: Path):
    monkeypatch.setattr(models_cmd, "_installed_tags", lambda cfg: ({}, "private:11435"))
    monkeypatch.setattr(models_cmd, "_residents", lambda cfg: {})
    info = ollama_server.ServerInfo(
        mode="private", base_url="http://127.0.0.1:21435", port=21435, pid=1, started_by_us=True
    )
    monkeypatch.setattr(models_cmd.ollama_server, "ensure_started", lambda cfg: info)
    monkeypatch.setattr(
        models_cmd.ollama_server,
        "api_tags",
        lambda port: [{"name": "gpt-oss:120b", "size": 65_000_000_000}],
    )
    monkeypatch.setattr(models_cmd.ollama_server, "pull_in_progress", lambda: None)
    monkeypatch.setattr(models_cmd.setup_flow, "registry_size", lambda tag: 20 * 1024**3)
    pulled: list[str] = []
    monkeypatch.setattr(
        models_cmd.setup_flow, "pull_model", lambda url, tag, console: pulled.append(tag) or True
    )
    result = runner.invoke(app, ["models", "--pull"])
    assert result.exit_code == 0
    # union across profiles, minus the already-installed reasoner
    assert pulled == ["qwen3.6:27b-mlx", "qwen3.6:35b-mlx"]


def _pull_env(monkeypatch: pytest.MonkeyPatch, registry_bytes: int | None):
    monkeypatch.setattr(models_cmd, "_installed_tags", lambda cfg: ({}, "private:11435"))
    monkeypatch.setattr(models_cmd, "_residents", lambda cfg: {})
    info = ollama_server.ServerInfo(
        mode="private", base_url="http://127.0.0.1:21435", port=21435, pid=1, started_by_us=True
    )
    monkeypatch.setattr(models_cmd.ollama_server, "ensure_started", lambda cfg: info)
    # all pulled except the reasoner, so exactly one >30GB tag is missing
    monkeypatch.setattr(
        models_cmd.ollama_server,
        "api_tags",
        lambda port: [{"name": t, "size": 1} for t in ("qwen3.6:35b-mlx", "qwen3.6:27b-mlx")],
    )
    monkeypatch.setattr(models_cmd.ollama_server, "pull_in_progress", lambda: None)
    monkeypatch.setattr(models_cmd.setup_flow, "registry_size", lambda tag: registry_bytes)
    pulled: list[str] = []
    monkeypatch.setattr(
        models_cmd.setup_flow, "pull_model", lambda url, tag, console: pulled.append(tag) or True
    )
    return pulled


def test_models_pull_over_30gb_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch, example_cfg, tmp_home: Path
):
    pulled = _pull_env(monkeypatch, 65 * 1024**3)
    result = runner.invoke(app, ["models", "--pull"], input="n\n")
    assert result.exit_code == 0
    assert pulled == [], "declining the >30GB confirm must skip the pull"
    assert "skipped" in result.output


def test_models_pull_yes_disarms_confirmation(
    monkeypatch: pytest.MonkeyPatch, example_cfg, tmp_home: Path
):
    pulled = _pull_env(monkeypatch, 65 * 1024**3)
    result = runner.invoke(app, ["models", "--pull", "--yes"])
    assert result.exit_code == 0
    assert pulled == ["gpt-oss:120b"]


def test_models_pull_respects_collision_guard(monkeypatch: pytest.MonkeyPatch, example_cfg, tmp_home: Path):
    monkeypatch.setattr(models_cmd, "_installed_tags", lambda cfg: ({}, "private:11435"))
    monkeypatch.setattr(models_cmd, "_residents", lambda cfg: {})
    info = ollama_server.ServerInfo(
        mode="private", base_url="http://127.0.0.1:21435", port=21435, pid=1, started_by_us=True
    )
    monkeypatch.setattr(models_cmd.ollama_server, "ensure_started", lambda cfg: info)
    monkeypatch.setattr(models_cmd.ollama_server, "api_tags", lambda port: [])
    monkeypatch.setattr(
        models_cmd.ollama_server, "pull_in_progress", lambda: "a pull is in progress elsewhere"
    )
    monkeypatch.setattr(models_cmd.setup_flow, "registry_size", lambda tag: 1024)
    monkeypatch.setattr(
        models_cmd.setup_flow,
        "pull_model",
        lambda *a, **k: pytest.fail("must not pull while another pull is in flight"),
    )
    result = runner.invoke(app, ["models", "--pull"])
    assert result.exit_code == 1
    assert "skipping" in result.output


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
