"""Memory, port, and neighbor-process awareness (Section 1.2 "memory etiquette").

MathScrambler shares this machine with a global Ollama and (sometimes) ComfyUI.
Before loading models we project our footprint, attribute current usage, and
refuse rather than push the machine past `memory.max_fraction` of physical RAM.
"""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass, field

import psutil

# Weights-on-disk to resident-memory multiplier: covers q8_0 KV cache at
# 8-16K num_ctx plus runner overhead. Tunable when bench data says otherwise.
FOOTPRINT_OVERHEAD = 1.15

COMFYUI_PORT = 8188


@dataclass(frozen=True)
class MemoryInfo:
    total: int
    available: int  # free + reclaimable (macOS "free" alone is deliberately tiny)
    pressure_level: int | None  # kern.memorystatus_vm_pressure_level: 1 normal / 2 warn / 4 critical


def memory_info() -> MemoryInfo:
    vm = psutil.virtual_memory()
    return MemoryInfo(total=vm.total, available=vm.available, pressure_level=_pressure_level())


def _pressure_level() -> int | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def find_free_port(start: int, host: str = "127.0.0.1", max_tries: int = 20) -> int:
    """First bindable port walking up from `start` (Section 1.2 port walking)."""
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port in {start}..{start + max_tries - 1}")


@dataclass(frozen=True)
class ComfyUIStatus:
    running: bool
    signals: list[str] = field(default_factory=list)


def detect_comfyui(port: int = COMFYUI_PORT) -> ComfyUIStatus:
    """Two independent signals: the default listener port, and a cmdline scan."""
    signals: list[str] = []
    if port_in_use(port):
        signals.append(f"listener on 127.0.0.1:{port}")
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "comfyui" in cmdline.lower():
            signals.append(f"process {proc.info['pid']}: {cmdline[:80]}")
            break
    return ComfyUIStatus(running=bool(signals), signals=signals)


def projected_footprint(weight_bytes: int) -> int:
    """Projected resident bytes for weights not yet loaded."""
    return int(weight_bytes * FOOTPRINT_OVERHEAD)


@dataclass(frozen=True)
class MemoryGate:
    ok: bool
    projected: int
    in_use: int
    limit: int
    total: int
    attribution: list[tuple[str, int]]  # (who, bytes) — shown verbatim on refusal
    message: str


def memory_gate(
    mem: MemoryInfo,
    projected: int,
    max_fraction: float,
    global_residents: list[tuple[str, int]],
    comfyui: ComfyUIStatus,
) -> MemoryGate:
    """Refuse when (memory in use + our projection) would exceed max_fraction of RAM.

    "In use" is total - available, so reclaimable cache doesn't count against us.
    On refusal the attribution names exactly what is using memory (Section 1.2).
    """
    in_use = mem.total - mem.available
    limit = int(mem.total * max_fraction)
    ok = in_use + projected <= limit

    attribution: list[tuple[str, int]] = list(global_residents)
    if comfyui.running:
        attribution.append(("ComfyUI (" + "; ".join(comfyui.signals) + ")", 0))
    accounted = sum(size for _, size in global_residents)
    attribution.append(("other (apps + OS, non-reclaimable)", max(0, in_use - accounted)))

    gb = 1024**3
    if ok:
        message = (
            f"memory ok: {in_use / gb:.1f} GB in use + {projected / gb:.1f} GB projected "
            f"<= {limit / gb:.1f} GB ({max_fraction:.0%} of {mem.total / gb:.0f} GB)"
        )
    else:
        who = ", ".join(
            f"{name}: {size / gb:.1f} GB" if size else name for name, size in attribution
        )
        message = (
            f"refusing to load models: {in_use / gb:.1f} GB in use + {projected / gb:.1f} GB projected "
            f"exceeds {limit / gb:.1f} GB ({max_fraction:.0%} of {mem.total / gb:.0f} GB). "
            f"Currently using memory — {who}. Try --lite (smaller models), or free memory and retry."
        )
    return MemoryGate(
        ok=ok,
        projected=projected,
        in_use=in_use,
        limit=limit,
        total=mem.total,
        attribution=attribution,
        message=message,
    )
