"""Running a blueprint's `solver_code` without trusting it (Section 3, Step B).

A blueprint's solver is model-written Python. It runs in a separate interpreter
(`_solver_child.py`) with restricted builtins, an import allowlist, no usable
stdout, and rlimits; this module owns the wall-clock timeout and the kill.

The session stays open across many solves on purpose. Rejection sampling calls
the solver once per candidate draw — hundreds of times for one variant — and a
fresh interpreter plus a sympy import per call would cost seconds each. One
child, many requests, killed on the first one that overruns.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict

CHILD = Path(__file__).with_name("_solver_child.py")

AnswerKind = Literal["integer", "rational", "float", "expression", "set", "text"]

# A solver that takes longer than this to load has a module-level loop or a very
# slow import; either way the wall-clock timeout for one solve is too short for it.
STARTUP_GRACE_S = 20.0


class SandboxError(RuntimeError):
    """The solver could not be loaded at all (syntax error, banned import, no solve())."""


class Answer(BaseModel):
    """One computed answer, in a form two of them can be compared by.

    `text` is canonical — `Fraction(6, 2)` and `6` both arrive as `integer` "6"
    — so equality is a string comparison for everything except expressions,
    which Step D compares with sympy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: AnswerKind
    text: str
    numeric: float | None = None
    items: list[str] | None = None

    @property
    def is_exact_number(self) -> bool:
        return self.kind in ("integer", "rational")


@dataclass(frozen=True)
class SolveResult:
    ok: bool
    answer: Answer | None = None
    error: str | None = None
    timed_out: bool = False
    duration_s: float = 0.0


def _encode_param(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"__fraction__": True, "n": value.numerator, "d": value.denominator}
    if isinstance(value, bool | int | float | str):
        return value
    raise SandboxError(f"parameter values must be numbers or strings, got {type(value).__name__}")


class SolverSession:
    """One loaded solver, callable many times. Use as a context manager."""

    def __init__(self, code: str, *, timeout_s: float = 5.0, memory_mb: int = 1024) -> None:
        self._code = code
        self._timeout_s = timeout_s
        self._memory_mb = memory_mb
        self._proc: subprocess.Popen[bytes] | None = None
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._buffer = b""
        self._dead_reason: str | None = None

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mathscrambler-solver-")
        tmp = self._tmp.name
        self._proc = subprocess.Popen(
            [sys.executable, "-I", str(CHILD)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp,
            # Scoped, like every other child this project spawns: the solver
            # inherits none of the caller's environment.
            env={"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp},
            start_new_session=True,
        )
        header = {
            "code": self._code,
            "memory_mb": self._memory_mb,
            # A CPU-time backstop for the whole session, well above any single
            # solve, so a solver that ignores signals still cannot spin forever.
            "cpu_seconds": max(30, int(self._timeout_s * 20)),
        }
        self._send(header)
        reply = self._read_reply(STARTUP_GRACE_S)
        if reply is None:
            # Distinguish "still thinking" from "already dead": a child that
            # cannot start at all would otherwise be reported as a 20s timeout,
            # which sends whoever reads it looking in entirely the wrong place.
            died = self._proc is not None and self._proc.poll() is not None
            detail = self._drain_stderr() if died else ""
            self.close()
            raise SandboxError(
                f"solver process exited before loading{f': {detail}' if detail else ''}"
                if died
                else f"solver did not load within {STARTUP_GRACE_S:.0f}s"
            )
        if not reply.get("ok"):
            self.close()
            raise SandboxError(str(reply.get("error", "solver failed to load")))

    def _drain_stderr(self, limit: int = 500) -> str:
        """The dead child's last words. Only safe to call once it has exited."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return ""
        with suppress(OSError, ValueError):
            return proc.stderr.read().decode(errors="replace").strip()[-limit:]
        return ""

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            with suppress(OSError, ValueError):
                if proc.stdin:
                    proc.stdin.close()
            self._kill(proc)
        if self._tmp is not None:
            with suppress(OSError):
                self._tmp.cleanup()
            self._tmp = None

    @staticmethod
    def _kill(proc: subprocess.Popen[bytes]) -> None:
        """Kill the child's whole group, then reap it.

        The group, because a solver that spawned something would otherwise leave
        it behind; the reap, because an unreaped child stays in the process table
        looking alive — the same trap that made `server stop` report a wedged
        server it had already killed.
        """
        if proc.poll() is None:
            with suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with suppress(OSError, ValueError):
            proc.wait(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            with suppress(OSError, ValueError):
                if stream:
                    stream.close()

    # ------------------------------------------------------------------ protocol

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise SandboxError("solver session is not running")
        proc.stdin.write((json.dumps(payload) + "\n").encode())
        proc.stdin.flush()

    def _read_reply(self, timeout_s: float) -> dict[str, Any] | None:
        """One response line, or None if it did not arrive in time."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        deadline = time.monotonic() + timeout_s
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    return None
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:  # the child exited without answering
                    return None
                self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------ solving

    def solve(self, params: Mapping[str, Any]) -> SolveResult:
        """Call `solve(**params)` in the child. Never raises for solver failures."""
        if self._dead_reason is not None:
            return SolveResult(ok=False, error=self._dead_reason)
        started = time.monotonic()
        try:
            self._send({"params": {k: _encode_param(v) for k, v in params.items()}})
        except (SandboxError, BrokenPipeError, OSError) as e:
            self._dead_reason = f"solver session ended: {e}"
            return SolveResult(ok=False, error=self._dead_reason, duration_s=time.monotonic() - started)

        reply = self._read_reply(self._timeout_s)
        elapsed = time.monotonic() - started
        if reply is None:
            # No response means the child is either spinning or gone. Either way
            # the session cannot be trusted to be at a line boundary again.
            timed_out = self._proc is not None and self._proc.poll() is None
            self._dead_reason = (
                f"solver exceeded the {self._timeout_s:g}s timeout"
                if timed_out
                else "solver process died without answering"
            )
            self.close()
            return SolveResult(ok=False, error=self._dead_reason, timed_out=timed_out, duration_s=elapsed)
        if not reply.get("ok"):
            return SolveResult(ok=False, error=str(reply.get("error", "solver failed")), duration_s=elapsed)
        return SolveResult(ok=True, answer=Answer.model_validate(reply["answer"]), duration_s=elapsed)


def run_solver(
    code: str,
    params: Mapping[str, Any],
    *,
    timeout_s: float = 5.0,
    memory_mb: int = 1024,
) -> SolveResult:
    """One-shot convenience. Prefer a `SolverSession` when solving repeatedly."""
    try:
        with SolverSession(code, timeout_s=timeout_s, memory_mb=memory_mb) as session:
            return session.solve(params)
    except SandboxError as e:
        return SolveResult(ok=False, error=str(e))
