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
import logging
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

log = logging.getLogger("mathscrambler.ollama_server")

MIN_OLLAMA_VERSION = (0, 19)
APP_BUNDLE_BINARY = Path("/Applications/Ollama.app/Contents/Resources/ollama")
PORT_WALK_TRIES = 20
LOG_ROTATE_BYTES = 10 * 1024 * 1024
CREATE_TIME_TOLERANCE_S = 1.0
SPAWN_MARKER = "--- mathscramble spawn "
OWNED_PROBE_RETRIES = 4  # extra health probes before declaring an owned server wedged
SHARED_PROBE_TTL_S = 24 * 3600


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
    """Full environment copy, foreign OLLAMA_* stripped, our settings overlaid.

    A full copy (not a minimal env) because ollama needs HOME to find the shared
    model store and PATH/TMPDIR for its runners. Any globally-injected OLLAMA_*
    (e.g. via launchctl setenv) is stripped so it cannot reconfigure OUR server —
    except OLLAMA_MODELS, which is deliberately inherited untouched so the store
    stays wherever the user's already is (Section 1.2). We never set it.
    """
    env = {
        k: v for k, v in os.environ.items() if not k.startswith("OLLAMA_") or k == "OLLAMA_MODELS"
    }
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

# Loopback probes must never route through a user proxy (HTTP_PROXY et al.).
_NO_PROXY_CLIENT_KW = {"trust_env": False}


def _get_json_dict(url: str, timeout: float) -> dict | None:
    try:
        resp = httpx.get(url, timeout=timeout, **_NO_PROXY_CLIENT_KW)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def api_version(port: int, timeout: float = 2.0) -> str | None:
    data = _get_json_dict(f"http://127.0.0.1:{port}/api/version", timeout)
    if data is None:
        return None
    version = data.get("version")
    return str(version) if version else None


def _models_list(port: int, endpoint: str, timeout: float) -> list[dict] | None:
    data = _get_json_dict(f"http://127.0.0.1:{port}/api/{endpoint}", timeout)
    if data is None:
        return None
    models = data.get("models", [])
    return [m for m in models if isinstance(m, dict)] if isinstance(models, list) else None


def api_ps(port: int, timeout: float = 3.0) -> list[dict] | None:
    """Resident models on a server, or None if unreachable/garbled (NOT the same as idle!)."""
    return _models_list(port, "ps", timeout)


def api_tags(port: int, timeout: float = 5.0) -> list[dict] | None:
    return _models_list(port, "tags", timeout)


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
    sizes: dict[Path, int] = {}
    for p in partials:
        with suppress(FileNotFoundError):  # completed/renamed between glob and stat
            sizes[p] = p.stat().st_size
    if not sizes:
        return None
    time.sleep(settle_s)
    for p, size in sizes.items():
        try:
            if p.stat().st_size != size:
                return f"a pull is in progress in the shared model store ({p.name} is growing)"
        except FileNotFoundError:
            continue  # completed between stats
    return f"{len(sizes)} partial blob(s) in the shared store (a stalled or in-flight pull); retry later"


# --------------------------------------------------------------------------- state file


def read_state() -> dict:
    """state.json: capability-probe results, shared-mode verification, misc caches."""
    try:
        data = json.loads(paths.state_file_path().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(update: dict) -> None:
    state = read_state()
    state.update(update)
    path = paths.state_file_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.rename(tmp, path)


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


def _tail_this_spawn(path: Path, lines: int = 20) -> str:
    """Last lines of the server log, scoped to the MOST RECENT spawn marker.

    Scoping matters: the log is append-mode, so an unscoped tail can surface a
    PREVIOUS run's error lines (e.g. an old bind failure) and misclassify this
    run's failure.
    """
    try:
        all_lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "(no server log available)"
    start = 0
    for i, line in enumerate(all_lines):
        if line.startswith(SPAWN_MARKER):
            start = i + 1
    return "\n".join(all_lines[start:][-lines:])


def _reap(pid: int) -> None:
    """Clear a dead child from the process table, if it is ours to clear."""
    with suppress(ChildProcessError, OSError):
        os.waitpid(pid, os.WNOHANG)


def _pid_alive(pid: int) -> bool:
    """Is `pid` a running process?

    A child we terminated but have not reaped is a zombie: really dead, still
    in the process table. `psutil.pid_exists` says True for it, so a caller
    that spawned the server itself — the normal case for `run`/`ui`, which
    start a server and stop it at the end — would watch its own successful
    kill time out and report the server as wedged.
    """
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # alive and someone else's; never claim we killed it


def _terminate_group(pid: int, term_wait_s: float) -> bool:
    """SIGTERM the process group, wait, escalate to SIGKILL. True if it's gone."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        _reap(pid)
        return True
    except PermissionError:
        return False
    deadline = time.monotonic() + term_wait_s
    while time.monotonic() < deadline:
        _reap(pid)  # if we are the parent, this turns a zombie into a gone pid
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    time.sleep(0.2)
    _reap(pid)
    return not _pid_alive(pid)


def _spawn(
    binary: BinaryInfo,
    port: int,
    config: Config,
    health_timeout_s: float,
    poll_interval_s: float,
) -> PidRecord:
    log_path = paths.ollama_log_path()
    _rotate_log(log_path)
    env = scoped_env(port, config)
    overlay = {k: env[k] for k in SCOPED_ENV_KEYS}
    with open(log_path, "a") as log_fh:
        log_fh.write(f"\n{SPAWN_MARKER}{datetime.now(UTC).isoformat()} port={port} ---\n")
        log_fh.flush()
        proc = subprocess.Popen(
            [str(binary.path), "serve"],
            env=env,
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
                f"log lines from this attempt ({log_path}):\n{_tail_this_spawn(log_path)}"
            )
        if api_version(port, timeout=1.0):
            break
        time.sleep(poll_interval_s)
    else:
        reaped = _terminate_group(proc.pid, term_wait_s=5.0)
        leak_note = "" if reaped else f" WARNING: pid {proc.pid} survived SIGKILL — check it manually."
        raise ServerError(
            f"private ollama server did not answer /api/version on port {port} "
            f"within {health_timeout_s:.0f}s; log lines from this attempt:\n"
            f"{_tail_this_spawn(log_path)}{leak_note}"
        )
    if proc.poll() is not None:
        # Something answered the port but our child is dead — a foreign process
        # won the bind race. Never adopt a process we didn't start.
        raise ServerError(
            f"port {port} is answering but our spawned server exited (code {proc.returncode}) — "
            f"a foreign process holds the port; log:\n{_tail_this_spawn(log_path)}"
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


def _probe_owned_server(record: PidRecord, retries: int = OWNED_PROBE_RETRIES) -> bool:
    """Health-probe a server we own, with patience: one transient timeout must not
    condemn a live process (it may be scanning the store or paging under load)."""
    for attempt in range(retries):
        if api_version(record.port, timeout=2.0 + attempt):
            return True
        if not owns_process(record):
            return False
        time.sleep(0.5)
    return False


def ensure_started(
    config: Config,
    health_timeout_s: float = 30.0,
    poll_interval_s: float = 0.25,
) -> ServerInfo:
    """Start (or reuse) the server the engine should talk to. Idempotent; flock-guarded."""
    with _server_lock():
        if config.ollama.mode == "shared":
            shared = _try_shared_mode(config)
            if shared is not None:
                return shared
            log.warning(
                "config asks for shared mode but the global server on %d is unavailable, busy, "
                "or unverified — falling back to a private server",
                config.ollama.global_port,
            )
        record = read_pid_record()
        if record is not None:
            if owns_process(record):
                if _probe_owned_server(record):
                    return ServerInfo(
                        mode="private",
                        base_url=f"http://127.0.0.1:{record.port}",
                        port=record.port,
                        pid=record.pid,
                        started_by_us=False,
                    )
                # Owned but wedged: stop the process we own before replacing it.
                # Never just unlink the record — that would orphan our own server.
                log.warning(
                    "private server pid %d is alive but not answering on port %d — stopping it "
                    "before starting a fresh one",
                    record.pid,
                    record.port,
                )
                if not _terminate_group(record.pid, term_wait_s=10.0):
                    raise ServerError(
                        f"private server pid {record.pid} is wedged and could not be stopped; "
                        f"inspect it manually before starting another (`ps -p {record.pid}`)"
                    )
                paths.pid_file_path().unlink(missing_ok=True)
            else:
                # Provably not our process (or gone): the record is stale. Clean, never signal.
                paths.pid_file_path().unlink(missing_ok=True)

        binary = discover_binary()
        if binary.version_tuple < MIN_OLLAMA_VERSION:
            raise ServerError(
                f"ollama {binary.version} at {binary.path} is older than required "
                f"{'.'.join(map(str, MIN_OLLAMA_VERSION))}; upgrade with `brew upgrade ollama`"
            )
        try:
            port = sysinfo.find_free_port(config.ollama.port, max_tries=PORT_WALK_TRIES)
        except RuntimeError as e:
            raise ServerError(str(e)) from e
        try:
            record = _spawn(binary, port, config, health_timeout_s, poll_interval_s)
        except ServerError as first_error:
            # The error's log tail is scoped to THIS spawn attempt (see _tail_this_spawn),
            # so this match cannot be triggered by a previous run's stale bind failure.
            if "address already in use" not in str(first_error):
                raise
            try:
                port = sysinfo.find_free_port(port + 1, max_tries=PORT_WALK_TRIES)
            except RuntimeError as e:
                raise ServerError(str(e)) from e
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

    Etiquette layered on top of the spec's probe, because the probe itself must
    never evict a neighbor's model:
    - a successful probe is cached in state.json (TTL 24h) so we don't reload
      probe models on every invocation — and so our own residents on the global
      server don't read as "busy" on the next run;
    - the probe only runs while the global server is verifiably idle, re-checked
      immediately before EACH load, and aborts the moment a foreign model appears;
    - an unreachable /api/ps means residency is UNKNOWN — treated as busy, not idle;
    - probe loads use keep_alive=0 so they unload immediately.
    A remaining sliver of check-vs-load race is unavoidable without a reservation
    API; the re-check-per-load plus tiny (<=2 GB) models keeps the worst case to
    evicting nothing warm (we only proceed from a verified-idle server).
    """
    port = config.ollama.global_port
    version = api_version(port)
    if version is None:
        return None

    state = read_state()
    verified_at = state.get("shared_verified_at")
    if verified_at:
        try:
            age = (datetime.now(UTC) - datetime.fromisoformat(verified_at)).total_seconds()
        except ValueError:
            age = SHARED_PROBE_TTL_S + 1
        if age < SHARED_PROBE_TTL_S:
            return ServerInfo(
                mode="shared", base_url=f"http://127.0.0.1:{port}", port=port, pid=None, started_by_us=False
            )

    residents = api_ps(port)
    if residents is None or residents:  # unknown counts as busy
        return None
    tags = api_tags(port) or []
    small = sorted(
        (m for m in tags if 0 < int(m.get("size", 0)) <= 2 * 1024**3),
        key=lambda m: int(m["size"]),
    )[:2]
    if len(small) < 2:
        return None
    probe_names = {m["name"] for m in small}
    try:
        for model in small:
            now_resident = api_ps(port)
            if now_resident is None or any(m.get("name") not in probe_names for m in now_resident):
                log.warning("global server became busy mid-probe — aborting shared-mode probe")
                return None
            httpx.post(
                f"http://127.0.0.1:{port}/api/generate",
                json={"model": model["name"], "keep_alive": 0},
                timeout=120,
                **_NO_PROXY_CLIENT_KW,
            ).raise_for_status()
        loaded = api_ps(port) or []
    except httpx.HTTPError:
        return None
    if len(loaded) >= 2:
        write_state({"shared_verified_at": datetime.now(UTC).isoformat(), "shared_version": version})
        return ServerInfo(
            mode="shared", base_url=f"http://127.0.0.1:{port}", port=port, pid=None, started_by_us=False
        )
    return None


@dataclass(frozen=True)
class StopResult:
    ok: bool  # the desired end state holds: no private server of ours is running
    stopped: bool  # we actually terminated a process
    reason: str
    pid: int | None = None


def stop(term_wait_s: float = 10.0) -> StopResult:
    """SIGTERM (then SIGKILL) the private server — only after proving we own it."""
    with _server_lock():
        record = read_pid_record()
        if record is None:
            return StopResult(ok=True, stopped=False, reason="no private server PID file — nothing to stop")
        if not owns_process(record):
            paths.pid_file_path().unlink(missing_ok=True)
            return StopResult(
                ok=True,
                stopped=False,
                reason=(
                    f"PID {record.pid} is not the server this tool started "
                    "(stale or recycled PID) — refusing to signal it; cleaned the stale PID file"
                ),
            )
        gone = _terminate_group(record.pid, term_wait_s)
        if not gone:
            return StopResult(
                ok=False,
                stopped=False,
                reason=f"could not stop private server pid {record.pid} (no permission or wedged)",
                pid=record.pid,
            )
        paths.pid_file_path().unlink(missing_ok=True)
        return StopResult(
            ok=True, stopped=True, reason=f"stopped private server (pid {record.pid})", pid=record.pid
        )


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
