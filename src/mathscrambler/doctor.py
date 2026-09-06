"""`mathscramble doctor` — green/yellow/red environment report (SPEC Section 1.3).

Pure data assembly; every red item carries the one command that fixes it.
The global server on 11434 is reported informationally only — its state is
never something doctor asks the user to change.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from rich.console import Console

from mathscrambler import ollama_server, paths, sysinfo
from mathscrambler.config import Config, ConfigError, load_config

MIN_FREE_DISK_GB = 50


class Level(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class Check:
    name: str
    level: Level
    detail: str
    fix: str | None = None  # required for RED


def _load_config_or_example() -> tuple[Config | None, Check]:
    try:
        return load_config(), Check("config", Level.GREEN, f"config.toml at {paths.config_path()}")
    except ConfigError as e:
        try:
            cfg = load_config(paths.config_example_path())
            return cfg, Check(
                "config",
                Level.YELLOW,
                f"{e} — using config.example.toml defaults for this report",
                fix="mathscramble setup",
            )
        except ConfigError as e2:
            return None, Check("config", Level.RED, str(e2), fix="mathscramble setup")


def _launchctl_getenv(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["launchctl", "getenv", name], capture_output=True, text=True, timeout=3, check=False
        )
        value = out.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def _check_binary() -> Check:
    try:
        binary = ollama_server.discover_binary()
    except ollama_server.ServerError as e:
        return Check("ollama binary", Level.RED, str(e), fix="brew install ollama")
    if binary.version_tuple < ollama_server.MIN_OLLAMA_VERSION:
        needed = ".".join(map(str, ollama_server.MIN_OLLAMA_VERSION))
        return Check(
            "ollama binary",
            Level.RED,
            f"{binary.path} is {binary.version}, need >= {needed}",
            fix="brew upgrade ollama",
        )
    return Check("ollama binary", Level.GREEN, f"{binary.path} (version {binary.version})")


def _check_global_server(cfg: Config | None) -> Check:
    port = cfg.ollama.global_port if cfg else 11434
    version = ollama_server.api_version(port)
    launchctl_value = _launchctl_getenv("OLLAMA_MAX_LOADED_MODELS") or "unset"
    if version is None:
        return Check(
            "global server (informational)",
            Level.YELLOW,
            f"no server on 127.0.0.1:{port} — fine; MathScrambler uses its own private server; "
            f"launchctl OLLAMA_MAX_LOADED_MODELS: {launchctl_value} (untouched by MathScrambler)",
        )
    residents = ollama_server.api_ps(port) or []
    names = ", ".join(m.get("name", "?") for m in residents) or "none"
    return Check(
        "global server (informational)",
        Level.GREEN,
        f"127.0.0.1:{port} ollama {version}; resident models: {names}; "
        f"launchctl OLLAMA_MAX_LOADED_MODELS: {launchctl_value} (untouched by MathScrambler)",
    )


def _check_private_server(cfg: Config | None) -> Check:
    if cfg and cfg.ollama.mode == "shared":
        return Check("private server", Level.GREEN, "config mode is 'shared' — no private server expected")
    st = ollama_server.status()
    if st.running and st.record is not None:
        env = ", ".join(f"{k.removeprefix('OLLAMA_')}={v}" for k, v in sorted(st.record.env.items()))
        return Check("private server", Level.GREEN, f"{st.detail}; spawned with {env}")
    return Check(
        "private server",
        Level.YELLOW,
        f"{st.detail} (started automatically by `ui`/`run` when needed)",
        fix="mathscramble server start",
    )


def _check_model_store() -> Check:
    store = paths.default_model_store()
    if not store.is_dir():
        return Check(
            "model store",
            Level.YELLOW,
            f"{store} does not exist yet (created by the first `ollama pull`)",
        )
    usage = shutil.disk_usage(store)
    free_gb = usage.free / 1024**3
    level = Level.GREEN if free_gb >= MIN_FREE_DISK_GB else Level.YELLOW
    detail = f"{store} (shared with the global server); {free_gb:.0f} GB free on its volume"
    pulling = ollama_server.pull_in_progress(store, settle_s=1.0)
    if pulling:
        detail += f"; note: {pulling}"
    return Check("model store", level, detail)


def _query_tags(cfg: Config | None) -> tuple[list[dict], str] | None:
    """Installed tags from whichever of our usable servers answers: (tags, source)."""
    st = ollama_server.status()
    if st.running and st.record is not None:
        tags = ollama_server.api_tags(st.record.port)
        if tags is not None:
            return tags, f"private:{st.record.port}"
    port = cfg.ollama.global_port if cfg else 11434
    tags = ollama_server.api_tags(port)
    if tags is not None:
        return tags, f"global:{port}"
    return None


def _check_roles(cfg: Config | None) -> list[Check]:
    if cfg is None:
        return []
    queried = _query_tags(cfg)
    if queried is None:
        return [
            Check(
                "model roles",
                Level.YELLOW,
                "no reachable Ollama server to list installed models",
                fix="mathscramble server start",
            )
        ]
    tags, source = queried
    installed = {m.get("name", ""): int(m.get("size", 0)) for m in tags}
    residents_private: set[str] = set()
    residents_global: set[str] = set()
    st = ollama_server.status()
    if st.running and st.residents:
        residents_private = {m.get("name", "") for m in st.residents}
    global_ps = ollama_server.api_ps(cfg.ollama.global_port if cfg else 11434) or []
    residents_global = {m.get("name", "") for m in global_ps}

    checks: list[Check] = []
    roles_table = cfg.active_roles()
    seen: set[str] = set()
    for role in ("vision", "reasoner", "fast"):
        rc = roles_table.resolve(role)  # type: ignore[arg-type]
        label = f"role {role}"
        alias = getattr(roles_table, role)
        if isinstance(alias, str):
            checks.append(Check(label, Level.GREEN, f"aliased to {alias} ({rc.tag})"))
            continue
        seen.add(rc.tag)
        mlx = "MLX" if "-mlx" in rc.tag else "non-MLX"
        where = (
            "resident on private"
            if rc.tag in residents_private
            else "resident on global"
            if rc.tag in residents_global
            else "not resident"
        )
        if rc.tag in installed:
            size_gb = installed[rc.tag] / 1e9
            checks.append(
                Check(label, Level.GREEN, f"{rc.tag} pulled ({size_gb:.1f} GB, {mlx}, {where}; via {source})")
            )
        else:
            checks.append(
                Check(label, Level.RED, f"{rc.tag} not pulled ({mlx})", fix=f"ollama pull {rc.tag}")
            )
    return checks


def _check_memory(cfg: Config | None) -> Check:
    mem = sysinfo.memory_info()
    gb = 1024**3
    if cfg is None:
        detail = f"{mem.total / gb:.0f} GB physical, {mem.available / gb:.0f} GB available"
        return Check("memory", Level.GREEN, detail)
    queried = _query_tags(cfg)
    installed = {m.get("name", ""): int(m.get("size", 0)) for m in (queried[0] if queried else [])}
    st = ollama_server.status()
    resident = {m.get("name", "") for m in (st.residents or [])} if st.running else set()
    tags_needed = set(cfg.active_roles().resolved_tags().values())
    weight_bytes = sum(installed.get(tag, 0) for tag in tags_needed if tag not in resident)
    projected = sysinfo.projected_footprint(weight_bytes)
    global_ps = ollama_server.api_ps(cfg.ollama.global_port) or []
    global_residents = [
        (f"global ollama: {m.get('name', '?')}", int(m.get("size_vram", m.get("size", 0)))) for m in global_ps
    ]
    gate = sysinfo.memory_gate(
        mem, projected, cfg.memory.max_fraction, global_residents, sysinfo.detect_comfyui()
    )
    unknown = [t for t in tags_needed if t not in installed and t not in resident]
    note = f" (un-pulled tags not counted: {', '.join(unknown)})" if unknown else ""
    if gate.ok:
        return Check("memory", Level.GREEN, gate.message + note)
    return Check("memory", Level.RED, gate.message + note, fix="mathscramble run ... --lite")


def _check_comfyui() -> Check:
    comfy = sysinfo.detect_comfyui()
    if comfy.running:
        return Check(
            "ComfyUI / GPU neighbors",
            Level.YELLOW,
            f"ComfyUI detected ({'; '.join(comfy.signals)}) — memory etiquette will account for it",
        )
    return Check("ComfyUI / GPU neighbors", Level.GREEN, "not running")


def _check_dashboard_port(cfg: Config | None) -> Check:
    port = cfg.dashboard.port if cfg else 8765
    if sysinfo.port_in_use(port):
        return Check(
            "dashboard port",
            Level.YELLOW,
            f"127.0.0.1:{port} is taken — `mathscramble ui` will walk to the next free port and print it",
        )
    return Check("dashboard port", Level.GREEN, f"127.0.0.1:{port} available")


def _uv_tool_bin_dir() -> Path:
    try:
        out = subprocess.run(
            ["uv", "tool", "dir", "--bin"], capture_output=True, text=True, timeout=5, check=True
        )
        return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return Path.home() / ".local" / "bin"


def _check_shim() -> Check:
    """Both console scripts must resolve: `mathscramble` AND the `ms` alias (SPEC Section 0)."""
    main_shim = shutil.which("mathscramble")
    if not main_shim:
        bin_dir = _uv_tool_bin_dir()
        return Check(
            "uv tool shim",
            Level.RED,
            f"`mathscramble` not on PATH (shim dir: {bin_dir})",
            fix=f'add to your shell profile yourself: export PATH="{bin_dir}:$PATH"',
        )
    ms_shim = shutil.which("ms")
    if not ms_shim:
        return Check(
            "uv tool shim",
            Level.RED,
            f"`mathscramble` on PATH ({main_shim}) but the `ms` alias is missing",
            fix="uv tool install -e . --reinstall",
        )
    if "mathscrambler" not in str(Path(ms_shim).resolve()):
        return Check(
            "uv tool shim",
            Level.YELLOW,
            f"`ms` on PATH resolves to a different tool ({ms_shim}) — use `mathscramble` instead",
        )
    return Check("uv tool shim", Level.GREEN, f"`mathscramble` and `ms` on PATH ({main_shim})")


def run_checks() -> list[Check]:
    cfg, config_check = _load_config_or_example()
    checks = [
        config_check,
        _check_binary(),
        _check_global_server(cfg),
        _check_private_server(cfg),
        _check_model_store(),
        *_check_roles(cfg),
        _check_memory(cfg),
        _check_comfyui(),
        _check_dashboard_port(cfg),
        _check_shim(),
    ]
    return checks


_ICONS: dict[Level, str] = {
    Level.GREEN: "[green]✓[/green]",
    Level.YELLOW: "[yellow]![/yellow]",
    Level.RED: "[red]✗[/red]",
}


def render(checks: list[Check], console: Console) -> Literal[0, 1]:
    for check in checks:
        console.print(f" {_ICONS[check.level]} [bold]{check.name}[/bold]: {check.detail}")
        if check.fix and check.level != Level.GREEN:
            console.print(f"     [dim]fix:[/dim] {check.fix}")
    reds = sum(1 for c in checks if c.level == Level.RED)
    yellows = sum(1 for c in checks if c.level == Level.YELLOW)
    summary = f"{len(checks)} checks: {len(checks) - reds - yellows} green, {yellows} yellow, {reds} red"
    console.print(f"\n [bold]{summary}[/bold]")
    return 1 if reds else 0
