"""Server-lifecycle tests — never touch a real Ollama (fakes + HTTP stub only)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mathscrambler import ollama_server, paths
from mathscrambler.config import load_config
from mathscrambler.ollama_server import (
    SCOPED_ENV_KEYS,
    BinaryInfo,
    PidRecord,
    ServerError,
    owns_process,
    read_pid_record,
    scoped_env,
    write_pid_record,
)


@pytest.fixture()
def cfg():
    return load_config(paths.config_example_path())


def _record(pid: int = 4242, port: int = 21435, create_time: float = 1000.0) -> PidRecord:
    return PidRecord(
        pid=pid,
        port=port,
        create_time=create_time,
        started_at="2026-09-06T00:00:00+00:00",
        binary="/opt/homebrew/bin/ollama",
        version="0.30.10",
        env={"OLLAMA_MAX_LOADED_MODELS": "3"},
    )


# --------------------------------------------------------------------- version parse


def test_parse_version_prefers_client_line():
    both = "Warning: could not connect\nclient version is 0.30.10\nollama version is 0.30.7\n"
    assert ollama_server._parse_version_output(both) == "0.30.10"
    assert ollama_server._parse_version_output("ollama version is 0.30.10\n") == "0.30.10"
    assert ollama_server._parse_version_output("garbage") is None


# --------------------------------------------------------------------- scoped env


def test_scoped_env_overlay_exact(cfg, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    before = dict(os.environ)
    env = scoped_env(21435, cfg)
    assert dict(os.environ) == before, "scoped_env must not mutate the process environment"
    assert env["OLLAMA_HOST"] == "127.0.0.1:21435"
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "3"
    assert env["OLLAMA_NUM_PARALLEL"] == "1"
    assert env["OLLAMA_KEEP_ALIVE"] == "30m"
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert "OLLAMA_MODELS" not in env, "the shared model store location must be inherited, never set"
    # nothing else OLLAMA-flavored is invented
    ollama_keys = {k for k in env if k.startswith("OLLAMA_")} - {
        k for k in os.environ if k.startswith("OLLAMA_")
    }
    assert ollama_keys == set(SCOPED_ENV_KEYS)


def test_scoped_env_inherits_preexisting_ollama_models(cfg, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLLAMA_MODELS", "/custom/store")
    env = scoped_env(21435, cfg)
    assert env["OLLAMA_MODELS"] == "/custom/store", "a user-exported store location is inherited untouched"


def test_scoped_env_strips_foreign_ollama_vars(cfg, monkeypatch: pytest.MonkeyPatch):
    """A launchctl-injected OLLAMA_* var must not reconfigure OUR server."""
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "2048")
    monkeypatch.setenv("OLLAMA_DEBUG", "1")
    env = scoped_env(21435, cfg)
    assert "OLLAMA_CONTEXT_LENGTH" not in env
    assert "OLLAMA_DEBUG" not in env
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "3"


# --------------------------------------------------------------------- PID file


def test_pid_record_round_trip(tmp_home: Path):
    record = _record()
    write_pid_record(record)
    assert read_pid_record() == record


def test_unreadable_pid_file_is_cleaned(tmp_home: Path):
    paths.ensure_app_dirs()
    paths.pid_file_path().write_text("not json {")
    assert read_pid_record() is None
    assert not paths.pid_file_path().exists()


def test_owns_process_requires_all_three_legs(monkeypatch: pytest.MonkeyPatch):
    record = _record(pid=999999, create_time=1000.0)

    class FakeProc:
        def __init__(self, pid):
            pass

        def create_time(self):
            return 1000.5  # within tolerance

        def cmdline(self):
            return ["/opt/homebrew/bin/ollama", "serve"]

    monkeypatch.setattr(ollama_server.psutil, "Process", FakeProc)
    assert owns_process(record)

    class WrongStart(FakeProc):
        def create_time(self):
            return 5000.0  # recycled PID

    monkeypatch.setattr(ollama_server.psutil, "Process", WrongStart)
    assert not owns_process(record)

    class WrongCmd(FakeProc):
        def cmdline(self):
            return ["/usr/bin/python3", "something_else.py"]

    monkeypatch.setattr(ollama_server.psutil, "Process", WrongCmd)
    assert not owns_process(record)

    def gone(pid):
        raise ollama_server.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(ollama_server.psutil, "Process", gone)
    assert not owns_process(record)


# --------------------------------------------------------------------- ensure_started


def test_ensure_started_reuses_healthy_server(tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch):
    record = _record()
    write_pid_record(record)
    monkeypatch.setattr(ollama_server, "owns_process", lambda r: True)
    monkeypatch.setattr(ollama_server, "api_version", lambda port, timeout=2.0: "0.30.10")
    spawned = []
    monkeypatch.setattr(ollama_server, "_spawn", lambda *a, **k: spawned.append(a))
    info = ollama_server.ensure_started(cfg)
    assert not info.started_by_us
    assert info.port == record.port
    assert info.base_url == f"http://127.0.0.1:{record.port}"
    assert spawned == []


def test_ensure_started_stops_owned_wedged_server_before_respawn(
    tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch
):
    """An owned-but-unhealthy server must be STOPPED, never orphaned by a bare
    PID-file unlink followed by a duplicate spawn (Phase-1 review, high)."""
    old = _record(pid=4242, port=21435)
    write_pid_record(old)
    monkeypatch.setattr(ollama_server, "owns_process", lambda r: True)
    monkeypatch.setattr(ollama_server, "_probe_owned_server", lambda r: False)
    terminated: list = []
    monkeypatch.setattr(
        ollama_server, "_terminate_group", lambda pid, term_wait_s: terminated.append(pid) or True
    )
    monkeypatch.setattr(
        ollama_server, "discover_binary", lambda: BinaryInfo(Path("/fake/ollama"), "0.30.10")
    )
    monkeypatch.setattr(ollama_server.sysinfo, "find_free_port", lambda start, max_tries=20: start)

    def fake_spawn(binary, port, config, health_timeout_s, poll_interval_s):
        new = _record(pid=777, port=port)
        write_pid_record(new)
        return new

    monkeypatch.setattr(ollama_server, "_spawn", fake_spawn)
    info = ollama_server.ensure_started(cfg)
    assert terminated == [4242], "the owned wedged server must be terminated before respawning"
    assert info.pid == 777


def test_ensure_started_raises_if_owned_wedged_server_unstoppable(
    tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch
):
    write_pid_record(_record(pid=4242))
    monkeypatch.setattr(ollama_server, "owns_process", lambda r: True)
    monkeypatch.setattr(ollama_server, "_probe_owned_server", lambda r: False)
    monkeypatch.setattr(ollama_server, "_terminate_group", lambda pid, term_wait_s: False)
    with pytest.raises(ServerError, match="wedged"):
        ollama_server.ensure_started(cfg)
    assert read_pid_record() is not None, "the record of a live owned process must be kept"


def test_ensure_started_cleans_stale_and_spawns_on_walked_port(
    tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch
):
    write_pid_record(_record(pid=1))  # stale: owns_process will fail
    monkeypatch.setattr(ollama_server, "owns_process", lambda r: False)
    monkeypatch.setattr(
        ollama_server, "discover_binary", lambda: BinaryInfo(Path("/fake/ollama"), "0.30.10")
    )
    monkeypatch.setattr(ollama_server.sysinfo, "find_free_port", lambda start, max_tries=20: start + 3)
    spawn_calls = []

    def fake_spawn(binary, port, config, health_timeout_s, poll_interval_s):
        record = _record(pid=777, port=port)
        write_pid_record(record)
        spawn_calls.append(port)
        return record

    monkeypatch.setattr(ollama_server, "_spawn", fake_spawn)
    info = ollama_server.ensure_started(cfg)
    assert info.started_by_us
    assert spawn_calls == [cfg.ollama.port + 3]
    assert info.port == cfg.ollama.port + 3
    assert read_pid_record().pid == 777


def test_ensure_started_rejects_old_binary(tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        ollama_server, "discover_binary", lambda: BinaryInfo(Path("/fake/ollama"), "0.18.9")
    )
    with pytest.raises(ServerError, match="older than required"):
        ollama_server.ensure_started(cfg)


# --------------------------------------------------------------------- _spawn health wait


class FakeProc:
    def __init__(self, pid: int = 555, returncode: int | None = None):
        self.pid = pid
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True


def test_spawn_raises_with_this_runs_log_when_child_dies(
    tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch
):
    paths.ensure_app_dirs()
    # A PREVIOUS session's bind failure sits in the append-mode log...
    paths.ollama_log_path().write_text("Error: listen tcp: bind: address already in use (STALE)\n")

    def fake_popen(cmd, env, stdout, stderr, cwd, start_new_session):
        stdout.write("Error: something else entirely went wrong\n")
        stdout.flush()
        return FakeProc(returncode=1)

    monkeypatch.setattr(ollama_server.subprocess, "Popen", fake_popen)
    binary = BinaryInfo(Path("/fake/ollama"), "0.30.10")
    with pytest.raises(ServerError) as exc:
        ollama_server._spawn(binary, 21435, cfg, health_timeout_s=1.0, poll_interval_s=0.01)
    # ...and must NOT leak into this run's error (it would misclassify the failure
    # as a bind race and trigger a bogus port-walk retry).
    assert "something else entirely" in str(exc.value)
    assert "STALE" not in str(exc.value)


def test_spawn_bind_race_error_carries_this_runs_bind_line(
    tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch
):
    paths.ensure_app_dirs()

    def fake_popen(cmd, env, stdout, stderr, cwd, start_new_session):
        stdout.write("Error: listen tcp 127.0.0.1:21435: bind: address already in use\n")
        stdout.flush()
        return FakeProc(returncode=1)

    monkeypatch.setattr(ollama_server.subprocess, "Popen", fake_popen)
    binary = BinaryInfo(Path("/fake/ollama"), "0.30.10")
    with pytest.raises(ServerError, match="address already in use"):
        ollama_server._spawn(binary, 21435, cfg, health_timeout_s=1.0, poll_interval_s=0.01)


def test_spawn_times_out_and_reaps_child(tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch):
    paths.ensure_app_dirs()
    proc = FakeProc(returncode=None)
    monkeypatch.setattr(ollama_server.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(ollama_server, "api_version", lambda port, timeout=2.0: None)
    reaped: list = []
    monkeypatch.setattr(
        ollama_server, "_terminate_group", lambda pid, term_wait_s: reaped.append(pid) or True
    )
    binary = BinaryInfo(Path("/fake/ollama"), "0.30.10")
    with pytest.raises(ServerError, match="did not answer"):
        ollama_server._spawn(binary, 21435, cfg, health_timeout_s=0.2, poll_interval_s=0.01)
    assert reaped == [proc.pid], "a health-timeout child must be killed via its process group"


def test_spawn_rejects_foreign_responder(tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch):
    """Port answers but our child died: a foreign process won the bind race —
    it must never be adopted as ours."""
    paths.ensure_app_dirs()
    proc = FakeProc(returncode=None)
    monkeypatch.setattr(ollama_server.subprocess, "Popen", lambda *a, **k: proc)

    def version_then_dead(port, timeout=2.0):
        proc.returncode = 1  # child dies exactly as something else answers
        return "9.9.9"

    monkeypatch.setattr(ollama_server, "api_version", version_then_dead)
    binary = BinaryInfo(Path("/fake/ollama"), "0.30.10")
    with pytest.raises(ServerError, match="foreign process"):
        ollama_server._spawn(binary, 21435, cfg, health_timeout_s=1.0, poll_interval_s=0.01)
    assert read_pid_record() is None


def test_spawn_records_scoped_env_and_pid(tmp_home: Path, cfg, monkeypatch: pytest.MonkeyPatch):
    paths.ensure_app_dirs()
    captured: dict = {}

    def fake_popen(cmd, env, stdout, stderr, cwd, start_new_session):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["start_new_session"] = start_new_session
        return FakeProc(pid=888)

    class FakePsProc:
        def __init__(self, pid):
            captured["ps_pid"] = pid

        def create_time(self):
            return 12345.0

    monkeypatch.setattr(ollama_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ollama_server, "api_version", lambda port, timeout=2.0: "0.30.10")
    monkeypatch.setattr(ollama_server.psutil, "Process", FakePsProc)
    record = ollama_server._spawn(
        BinaryInfo(Path("/fake/ollama"), "0.30.10"), 21440, cfg, health_timeout_s=1.0, poll_interval_s=0.01
    )
    assert captured["cmd"] == ["/fake/ollama", "serve"]
    assert captured["start_new_session"] is True
    assert captured["env"]["OLLAMA_HOST"] == "127.0.0.1:21440"
    assert record.pid == 888
    assert record.port == 21440
    assert record.create_time == 12345.0
    assert record.env == {k: captured["env"][k] for k in SCOPED_ENV_KEYS}
    assert read_pid_record() == record


# --------------------------------------------------------------------- stop


def test_stop_refuses_unowned_pid(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    write_pid_record(_record(pid=12345))
    monkeypatch.setattr(ollama_server, "owns_process", lambda r: False)
    kills = []
    monkeypatch.setattr(ollama_server.os, "killpg", lambda *a: kills.append(a))
    result = ollama_server.stop()
    assert not result.stopped
    assert result.ok, "end state (no server of ours) holds, so ok must be True"
    assert "refusing to signal" in result.reason
    assert kills == []
    assert read_pid_record() is None  # stale file cleaned


def test_stop_terminates_owned_process(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    write_pid_record(_record(pid=54321))
    monkeypatch.setattr(ollama_server, "owns_process", lambda r: True)
    kills: list = []
    monkeypatch.setattr(ollama_server.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(ollama_server.psutil, "pid_exists", lambda pid: False)
    result = ollama_server.stop()
    assert result.stopped
    assert result.ok
    assert result.pid == 54321
    assert kills == [(54321, ollama_server.signal.SIGTERM)]
    assert read_pid_record() is None


def test_stop_when_process_already_gone_is_ok(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    """A server that died on its own: nothing to signal, but the end state holds."""
    write_pid_record(_record(pid=54321))
    monkeypatch.setattr(ollama_server, "owns_process", lambda r: True)

    def killpg_gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(ollama_server.os, "killpg", killpg_gone)
    result = ollama_server.stop()
    assert result.ok
    assert read_pid_record() is None


def test_stop_unstoppable_process_is_not_ok(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    write_pid_record(_record(pid=54321))
    monkeypatch.setattr(ollama_server, "owns_process", lambda r: True)
    monkeypatch.setattr(ollama_server, "_terminate_group", lambda pid, term_wait_s: False)
    result = ollama_server.stop()
    assert not result.ok
    assert read_pid_record() is not None, "keep the record of a process we could not stop"


# --------------------------------------------------------------------- probes over stub


def test_api_probes_against_stub(http_stub):
    http_stub.route("/api/version", {"version": "0.30.10"})
    http_stub.route("/api/ps", {"models": [{"name": "tiny:1b", "size": 1}]})
    http_stub.route("/api/tags", {"models": [{"name": "tiny:1b", "size": 1}]})
    assert ollama_server.api_version(http_stub.port) == "0.30.10"
    assert ollama_server.api_ps(http_stub.port) == [{"name": "tiny:1b", "size": 1}]
    assert ollama_server.api_tags(http_stub.port) == [{"name": "tiny:1b", "size": 1}]
    assert ollama_server.api_version(1) is None  # nothing listens on port 1


def test_api_probes_reject_garbled_responders(http_stub):
    """A foreign, non-Ollama listener must never pass a health check."""
    http_stub.route("/api/version", {})  # 200 but no version key
    http_stub.route("/api/ps", {"models": "not-a-list"})
    assert ollama_server.api_version(http_stub.port) is None
    assert ollama_server.api_ps(http_stub.port) is None


# --------------------------------------------------------------------- pull guard


def test_pull_in_progress_detects_growing_partial(tmp_path: Path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    partial = blobs / "sha256-abc-partial-0"
    partial.write_bytes(b"x" * 10)

    import threading

    def grow():
        partial.write_bytes(b"x" * 200)

    timer = threading.Timer(0.1, grow)
    timer.start()
    try:
        reason = ollama_server.pull_in_progress(store=tmp_path, settle_s=0.3)
    finally:
        timer.cancel()
    assert reason is not None
    assert "growing" in reason


def test_pull_in_progress_static_partial_still_blocks(tmp_path: Path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "sha256-abc-partial-0").write_bytes(b"x" * 10)
    reason = ollama_server.pull_in_progress(store=tmp_path, settle_s=0.05)
    assert reason is not None
    assert "partial" in reason


def test_pull_in_progress_clean_store(tmp_path: Path):
    (tmp_path / "blobs").mkdir()
    assert ollama_server.pull_in_progress(store=tmp_path, settle_s=0.05) is None
