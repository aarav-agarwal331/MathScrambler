"""Private Ollama server lifecycle (SPEC Section 1.2).

MathScrambler never touches the user's global Ollama server on 11434. It spawns
its own `ollama serve` on a dedicated loopback port with all OLLAMA_* settings
scoped to that child's environment, shares the global model store (OLLAMA_MODELS
deliberately untouched), and only ever signals a process it can prove it started:
pid + process create-time + cmdline + a health check on the recorded port must
all match the PID file before any signal is sent.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psutil

from mathscrambler import paths, sysinfo
from mathscrambler.config import Config

MIN_OLLAMA_VERSION = (0, 19)
APP_BUNDLE_BINARY = Path("/Applications/Ollama.app/Contents/Resources/ollama")
PORT_WALK_TRIES = 20
LOG_ROTATE_BYTES = 10 * 1024 * 1024
CREATE_TIME_TOLERANCE_S = 1.0


class ServerError(RuntimeError):
    pass


# --------------------------------------------------------------------------- binary


@dataclass(frozen=True)
class BinaryInfo:
    path: Path
    version: str

    @property
    def version_tuple(self) -> tuple[int, ...]:
        return tuple(int(p) for p in self.version.split("."))


def discover_binary() -> BinaryInfo:
    """The `ollama` binary to spawn: PATH first (realpath'd), then the app bundle."""
    candidates: list[Path] = []
    which = shutil.which("ollama")
    if which:
        candidates.append(Path(which).resolve())
    if APP_BUNDLE_BINARY.is_file():
        candidates.append(APP_BUNDLE_BINARY)
    for candidate in candidates:
        version = _binary_version(candidate)
        if version:
            return BinaryInfo(path=candidate, version=version)
    raise ServerError(
        "no working `ollama` binary found — install Ollama first: `brew install ollama` "
        "or download Ollama.app from https://ollama.com/download"
    )


def _binary_version(binary: Path) -> str | None:
    # Point OLLAMA_HOST at a dead port: with a reachable server, `ollama --version`
    # reports the SERVER's version, not this binary's.
    env = dict(os.environ)
    env["OLLAMA_HOST"] = "127.0.0.1:1"
    try:
        out = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=10, check=True, env=env
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_version_output(out.stdout + out.stderr)


def _parse_version_output(text: str) -> str | None:
    match = re.search(r"client version is (\d+(?:\.\d+)+)", text) or re.search(
        r"version is (\d+(?:\.\d+)+)", text
    )
    return match.group(1) if match else None


# --------------------------------------------------------------------------- scoped env


def scoped_env(port: int, config: Config) -> dict[str, str]:
    """Full environment copy overlaid with exactly the private server's settings.

    A full copy (not a minimal env) because ollama needs HOME to find the shared
    model store and PATH/TMPDIR for its runners. OLLAMA_MODELS is never set or
    unset here — the store stays wherever the user's already is (Section 1.2).
    """
    env = dict(os.environ)
    env.update(
        {
            "OLLAMA_HOST": f"127.0.0.1:{port}",
            "OLLAMA_MAX_LOADED_MODELS": str(config.ollama.max_loaded_models),
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_KEEP_ALIVE": config.ollama.keep_alive,
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
        }
    )
    return env


SCOPED_ENV_KEYS = (
    "OLLAMA_HOST",
    "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_KEEP_ALIVE",
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE",
)


# --------------------------------------------------------------------------- PID file


@dataclass(frozen=True)
class PidRecord:
    pid: int
    port: int
    create_time: float  # psutil process create time — defeats PID reuse
    started_at: str  # iso8601, informational
    binary: str
    version: str
    env: dict[str, str]  # the scoped OLLAMA_* overlay, for doctor / criterion 1


def read_pid_record() -> PidRecord | None:
    path = paths.pid_file_path()
    try:
        data = json.loads(path.read_text())
        return PidRecord(**data)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, TypeError, KeyError):
        # Unreadable PID file: treat as stale, never guess at a PID to signal.
        path.unlink(missing_ok=True)
        return None


def write_pid_record(record: PidRecord) -> None:
    path = paths.pid_file_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(record), indent=2))
    os.rename(tmp, path)


def owns_process(record: PidRecord) -> bool:
    """The triple that gates every signal: alive + create-time match + ollama serve cmdline."""
    try:
        proc = psutil.Process(record.pid)
        if abs(proc.create_time() - record.create_time) > CREATE_TIME_TOLERANCE_S:
            return False
        cmdline = " ".join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return "ollama" in cmdline and "serve" in cmdline


# --------------------------------------------------------------------------- HTTP probes


def api_version(port: int, timeout: float = 2.0) -> str | None:
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/api/version", timeout=timeout)
        resp.raise_for_status()
        return str(resp.json().get("version"))
    except (httpx.HTTPError, ValueError):
        return None


def api_ps(port: int, timeout: float = 3.0) -> list[dict] | None:
    """Resident models on a server, or None if unreachable."""
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/api/ps", timeout=timeout)
        resp.raise_for_status()
        return list(resp.json().get("models", []))
    except (httpx.HTTPError, ValueError):
        return None


def api_tags(port: int, timeout: float = 5.0) -> list[dict] | None:
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return list(resp.json().get("models", []))
    except (httpx.HTTPError, ValueError):
        return None


def pull_in_progress(store: Path | None = None, settle_s: float = 3.0) -> str | None:
    """Detect an in-flight pull in the shared store (concurrent pulls corrupt blobs).

    /api/ps does NOT show pulls; the reliable signal is a growing ``*-partial*``
    blob. Returns a human-readable reason, or None when it's safe to pull.
    """
    blobs = (store or paths.default_model_store()) / "blobs"
    if not blobs.is_dir():
        return None
    partials = list(blobs.glob("*-partial*"))
    if not partials:
        return None
    sizes = {p: p.stat().st_size for p in partials if p.exists()}
    time.sleep(settle_s)
    for p, size in sizes.items():
        try:
            if p.stat().st_size != size:
                return f"a pull is in progress in the shared model store ({p.name} is growing)"
        except FileNotFoundError:
            continue  # completed between stats
    return f"{len(partials)} partial blob(s) in the shared store (a stalled or in-flight pull); retry later"


# --------------------------------------------------------------------------- lifecycle


@dataclass(frozen=True)
class ServerInfo:
    mode: str  # "private" | "shared"
    base_url: str
    port: int
    pid: int | None  # None in shared mode
    started_by_us: bool


@contextmanager
def _server_lock() -> Iterator[None]:
    """Exclusive flock closing the check-then-spawn race between concurrent invocations."""
    paths.ensure_app_dirs()
    lock_path = paths.server_lock_path()
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _rotate_log(log_path: Path) -> None:
    try:
        if log_path.is_file() and log_path.stat().st_size > LOG_ROTATE_BYTES:
            log_path.replace(log_path.with_suffix(".log.1"))
    except OSError:
        pass


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no server log available)"


def _spawn(
    binary: BinaryInfo,
    port: int,
    config: Config,
    health_timeout_s: float,
    poll_interval_s: float,
) -> PidRecord:
    log_path = paths.ollama_log_path()
    _rotate_log(log_path)
    overlay = {k: v for k, v in scoped_env(port, config).items() if k in SCOPED_ENV_KEYS}
    with open(log_path, "a") as log_fh:
        log_fh.write(f"\n--- mathscramble spawn {datetime.now(UTC).isoformat()} port={port} ---\n")
        log_fh.flush()
        proc = subprocess.Popen(
            [str(binary.path), "serve"],
            env=scoped_env(port, config),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=paths.app_support_dir(),
            start_new_session=True,  # own process group: CLI Ctrl-C can't kill it implicitly
        )
    deadline = time.monotonic() + health_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ServerError(
                f"private ollama server exited during startup (code {proc.returncode}); "
                f"last log lines from {log_path}:\n{_tail(log_path)}"
            )
        if api_version(port, timeout=1.0):
            break
        time.sleep(poll_interval_s)
    else:
        proc.terminate()
        raise ServerError(
            f"private ollama server did not answer /api/version on port {port} "
            f"within {health_timeout_s:.0f}s; last log lines:\n{_tail(log_path)}"
        )
    try:
        create_time = psutil.Process(proc.pid).create_time()
    except psutil.NoSuchProcess as e:  # died right after health check — extremely unlikely
        raise ServerError("private ollama server died immediately after startup") from e
    record = PidRecord(
        pid=proc.pid,
        port=port,
        create_time=create_time,
        started_at=datetime.now(UTC).isoformat(),
        binary=str(binary.path),
        version=binary.version,
        env=overlay,
    )
    write_pid_record(record)
    return record


def ensure_started(
    config: Config,
    health_timeout_s: float = 30.0,
    poll_interval_s: float = 0.25,
) -> ServerInfo:
    """Start (or reuse) the server the engine should talk to. Idempotent; flock-guarded."""
    if config.ollama.mode == "shared":
        shared = _try_shared_mode(config)
        if shared is not None:
            return shared
        # fall through to private with a spawned server
    with _server_lock():
        record = read_pid_record()
        if record is not None:
            if owns_process(record) and api_version(record.port):
                return ServerInfo(
                    mode="private",
                    base_url=f"http://127.0.0.1:{record.port}",
                    port=record.port,
                    pid=record.pid,
                    started_by_us=False,
                )
            paths.pid_file_path().unlink(missing_ok=True)  # stale: clean, never signal

        binary = discover_binary()
        if binary.version_tuple < MIN_OLLAMA_VERSION:
            raise ServerError(
                f"ollama {binary.version} at {binary.path} is older than required "
                f"{'.'.join(map(str, MIN_OLLAMA_VERSION))}; upgrade with `brew upgrade ollama`"
            )
        port = sysinfo.find_free_port(config.ollama.port, max_tries=PORT_WALK_TRIES)
        try:
            record = _spawn(binary, port, config, health_timeout_s, poll_interval_s)
        except ServerError as first_error:
            if "address already in use" not in str(first_error):
                raise
            # lost a bind race with a foreign process: walk once more
            port = sysinfo.find_free_port(port + 1, max_tries=PORT_WALK_TRIES)
            record = _spawn(binary, port, config, health_timeout_s, poll_interval_s)
        return ServerInfo(
            mode="private",
            base_url=f"http://127.0.0.1:{record.port}",
            port=record.port,
            pid=record.pid,
            started_by_us=True,
        )


def _try_shared_mode(config: Config) -> ServerInfo | None:
    """Reuse the global server only when provably friendly (Section 1.2).

    The spec's probe loads two tiny models on the global server — but on a
    limit-1 server that would itself evict a resident model, violating "never
    evict another server's models". So the probe only runs when the global
    server is idle; otherwise we warn and fall back to private.
    """
    port = config.ollama.global_port
    if api_version(port) is None:
        return None
    residents = api_ps(port) or []
    if residents:
        return None  # cannot probe without risking eviction; caller falls back to private
    tags = api_tags(port) or []
    small = sorted(
        (m for m in tags if 0 < int(m.get("size", 0)) <= 2 * 1024**3),
        key=lambda m: int(m["size"]),
    )[:2]
    if len(small) < 2:
        return None
    try:
        for model in small:
            httpx.post(
                f"http://127.0.0.1:{port}/api/generate",
                json={"model": model["name"], "keep_alive": "10s"},
                timeout=120,
            ).raise_for_status()
        loaded = api_ps(port) or []
    except httpx.HTTPError:
        return None
    if len(loaded) >= 2:
        return ServerInfo(
            mode="shared", base_url=f"http://127.0.0.1:{port}", port=port, pid=None, started_by_us=False
        )
    return None


@dataclass(frozen=True)
class StopResult:
    stopped: bool
    reason: str
    pid: int | None = None


def stop(term_wait_s: float = 10.0) -> StopResult:
    """SIGTERM (then SIGKILL) the private server — only after proving we own it."""
    with _server_lock():
        record = read_pid_record()
        if record is None:
            return StopResult(stopped=False, reason="no private server PID file — nothing to stop")
        if not owns_process(record):
            paths.pid_file_path().unlink(missing_ok=True)
            return StopResult(
                stopped=False,
                reason=(
                    f"PID {record.pid} is not the server this tool started "
                    "(stale or recycled PID) — refusing to signal it; cleaned the stale PID file"
                ),
            )
        try:
            os.killpg(record.pid, signal.SIGTERM)
        except ProcessLookupError:
            paths.pid_file_path().unlink(missing_ok=True)
            return StopResult(stopped=False, reason="server already gone; cleaned the PID file")
        except PermissionError:
            return StopResult(
                stopped=False, reason=f"no permission to signal PID {record.pid} — refusing"
            )
        deadline = time.monotonic() + term_wait_s
        while time.monotonic() < deadline:
            if not psutil.pid_exists(record.pid):
                break
            time.sleep(0.1)
        else:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(record.pid, signal.SIGKILL)
        paths.pid_file_path().unlink(missing_ok=True)
        return StopResult(stopped=True, reason=f"stopped private server (pid {record.pid})", pid=record.pid)


@dataclass(frozen=True)
class ServerStatus:
    running: bool
    detail: str
    record: PidRecord | None = None
    api_version: str | None = None
    residents: list[dict] | None = None


def status() -> ServerStatus:
    record = read_pid_record()
    if record is None:
        return ServerStatus(running=False, detail="private server not running (no PID file)")
    if not owns_process(record):
        return ServerStatus(
            running=False,
            detail=f"stale PID file (pid {record.pid} is gone or not ours) — will be cleaned on next start",
            record=record,
        )
    version = api_version(record.port)
    if version is None:
        return ServerStatus(
            running=False,
            detail=f"pid {record.pid} alive but port {record.port} not answering — orphaned? "
            "`mathscramble server stop` will clean it up",
            record=record,
        )
    return ServerStatus(
        running=True,
        detail=f"private server pid {record.pid} on 127.0.0.1:{record.port} (ollama {version})",
        record=record,
        api_version=version,
        residents=api_ps(record.port),
    )
