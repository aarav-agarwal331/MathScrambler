from __future__ import annotations

import socket

import pytest

from mathscrambler import sysinfo


def test_find_free_port_walks_past_a_bound_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        taken = blocker.getsockname()[1]
        blocker.listen(1)
        free = sysinfo.find_free_port(taken, max_tries=5)
        assert free != taken
        assert free > taken
        assert sysinfo.port_in_use(taken)
    assert not sysinfo.port_in_use(free)


def test_find_free_port_exhaustion():
    # bind an ephemeral port then ask for exactly that one with max_tries=1
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        taken = blocker.getsockname()[1]
        blocker.listen(1)
        with pytest.raises(RuntimeError, match="no free port"):
            sysinfo.find_free_port(taken, max_tries=1)


def test_projected_footprint_applies_overhead():
    assert sysinfo.projected_footprint(100) == int(100 * sysinfo.FOOTPRINT_OVERHEAD)
    assert sysinfo.projected_footprint(0) == 0


def _mem(total: int, available: int) -> sysinfo.MemoryInfo:
    return sysinfo.MemoryInfo(total=total, available=available, pressure_level=1)


def test_memory_gate_ok_at_boundary():
    gb = 1024**3
    mem = _mem(total=100 * gb, available=60 * gb)  # 40 in use
    gate = sysinfo.memory_gate(
        mem, projected=45 * gb, max_fraction=0.85, global_residents=[], comfyui=sysinfo.ComfyUIStatus(False)
    )
    assert gate.ok  # 40 + 45 = 85 == limit
    assert "memory ok" in gate.message


def test_memory_gate_refusal_names_consumers():
    gb = 1024**3
    mem = _mem(total=100 * gb, available=30 * gb)  # 70 in use
    gate = sysinfo.memory_gate(
        mem,
        projected=30 * gb,
        max_fraction=0.85,
        global_residents=[("global ollama: llama3.3:70b", 42 * gb)],
        comfyui=sysinfo.ComfyUIStatus(True, ["listener on 127.0.0.1:8188"]),
    )
    assert not gate.ok
    assert "llama3.3:70b" in gate.message
    assert "ComfyUI" in gate.message
    assert "--lite" in gate.message
    assert "other" in gate.message


class _FakeProc:
    def __init__(self, pid: int, cmdline: list[str]):
        self.info = {"pid": pid, "cmdline": cmdline}


def test_detect_comfyui_via_cmdline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sysinfo.psutil,
        "process_iter",
        lambda attrs: [_FakeProc(1, ["python"]), _FakeProc(2, ["python", "/x/ComfyUI/main.py"])],
    )
    status = sysinfo.detect_comfyui(port=1)  # port 1: nothing listens there
    assert status.running
    assert any("ComfyUI/main.py" in s for s in status.signals)


def test_detect_comfyui_clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sysinfo.psutil, "process_iter", lambda attrs: [_FakeProc(1, ["python"])])
    assert not sysinfo.detect_comfyui(port=1).running


def test_detect_comfyui_ignores_processes_that_merely_mention_it(monkeypatch: pytest.MonkeyPatch):
    """An agent/editor session opened on the ComfyUI folder is not ComfyUI (live false positive)."""
    monkeypatch.setattr(
        sysinfo.psutil,
        "process_iter",
        lambda attrs: [
            _FakeProc(1, ["/Applications/Claude.app/Contents/MacOS/claude", "--add-dir", "/u/ComfyUI"]),
            _FakeProc(2, ["/bin/zsh", "-c", "cd ~/Claude_ComfyUI && ls"]),
            _FakeProc(3, ["/u/Claude_ComfyUI/.venv/bin/python", "-c", "print(1)"]),
        ],
    )
    assert not sysinfo.detect_comfyui(port=1).running
    monkeypatch.setattr(
        sysinfo.psutil,
        "process_iter",
        lambda attrs: [
            _FakeProc(4, ["/u/ComfyUI/.venv/bin/python3.12", "/u/ComfyUI/main.py", "--listen"]),
        ],
    )
    assert sysinfo.detect_comfyui(port=1).running


def test_detect_comfyui_via_port(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sysinfo.psutil, "process_iter", lambda attrs: [])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        status = sysinfo.detect_comfyui(port=port)
        assert status.running
        assert any("listener" in s for s in status.signals)


def test_memory_info_smoke():
    mem = sysinfo.memory_info()
    assert mem.total > 0
    assert 0 < mem.available <= mem.total
