"""The subprocess that actually runs `solver_code`. Never imported — run by path.

`mathscrambler.sandbox` spawns this with `python -I <this file>` and talks to it
over stdin/stdout in newline-delimited JSON: one request per line, one response
per line. It stays alive across many `solve()` calls because rejection sampling
needs hundreds of them and a fresh interpreter (plus a sympy import) per call
would cost seconds each.

What this is and is not: a blast radius, not a security boundary. Model-written
code is careless far more often than hostile, and the containment here — a
separate address space, an import allowlist, restricted builtins, no writable
stdout, CPU and address-space rlimits, and a parent that kills on wall-clock —
stops the careless cases and the obvious hostile ones. It does not stop a
determined attacker with native-code tricks. Nothing here is a substitute for
not running code from an untrusted source.
"""

from __future__ import annotations

import builtins
import json
import os
import resource
import sys
from fractions import Fraction
from typing import Any

# Enough to do mathematics with, and nothing that reaches the filesystem, the
# network, or another process. sympy submodules are matched by prefix.
ALLOWED_MODULES = frozenset(
    {
        "cmath",
        "collections",
        "collections.abc",
        "decimal",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "math",
        "numbers",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "sympy",
        "typing",
    }
)

# Removed from the solver's builtins. `open` and `__import__` are the doors out;
# `exec`/`eval`/`compile` re-open them; the introspection names are how you get
# back to the real builtins through an object's `__class__` chain.
DENIED_BUILTINS = frozenset(
    {
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "exit",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "vars",
        "__import__",
    }
)

MAX_TEXT = 4000  # a runaway expression must not become an unbounded response line


class SolverDenied(ImportError):
    pass


def _guarded_import(name: str, globals_=None, locals_=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if name in ALLOWED_MODULES or root in ALLOWED_MODULES:
        return _real_import(name, globals_, locals_, fromlist, level)
    raise SolverDenied(f"solver_code may not import {name!r} (allowed: {', '.join(sorted(ALLOWED_MODULES))})")


_real_import = builtins.__import__


def _restricted_builtins() -> dict[str, Any]:
    safe = {k: v for k, v in vars(builtins).items() if k not in DENIED_BUILTINS}
    safe["__import__"] = _guarded_import
    return safe


def _apply_limits(memory_mb: int, cpu_seconds: int) -> None:
    """Backstops for the parent's wall-clock kill, best-effort by design.

    RLIMIT_AS is honoured unevenly on macOS and RLIMIT_CPU only fires on CPU
    time, so neither replaces the parent's timeout — they catch the cases the
    parent's kill is slowest at, like a runaway allocation.
    """
    for limit, value in ((resource.RLIMIT_AS, memory_mb * 1024 * 1024), (resource.RLIMIT_CPU, cpu_seconds)):
        try:
            _soft, hard = resource.getrlimit(limit)
            ceiling = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(limit, (ceiling, hard))
        except (ValueError, OSError):
            pass


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__fraction__"):
        return Fraction(int(value["n"]), int(value["d"]))
    return value


def _encode_number(value: Any) -> dict[str, Any] | None:
    if isinstance(value, bool):
        return {"kind": "integer", "text": str(int(value)), "numeric": float(value)}
    if isinstance(value, int):
        return {"kind": "integer", "text": str(value), "numeric": float(value)}
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return {"kind": "integer", "text": str(value.numerator), "numeric": float(value)}
        return {"kind": "rational", "text": f"{value.numerator}/{value.denominator}", "numeric": float(value)}
    if isinstance(value, float):
        return {"kind": "float", "text": repr(value), "numeric": value}
    return None


def _encode(value: Any) -> dict[str, Any]:
    """One answer as the parent's `Answer` shape.

    Integer-valued rationals and sympy Integers collapse to `integer`: whether a
    solver happened to return `Fraction(6, 2)` or `6` is an implementation
    detail, and letting it through would make two identical answers compare
    unequal and look like a verification failure.
    """
    encoded = _encode_number(value)
    if encoded is not None:
        return encoded

    sympy = sys.modules.get("sympy")
    if sympy is not None and isinstance(value, sympy.Basic):
        if value.is_Integer:
            return {"kind": "integer", "text": str(int(value)), "numeric": float(value)}
        if value.is_Rational:
            return {
                "kind": "rational",
                "text": f"{value.p}/{value.q}",
                "numeric": float(value),
            }
        if value.is_Number and value.is_real:
            return {"kind": "float", "text": str(value), "numeric": float(value)}
        if isinstance(value, sympy.Set | frozenset):
            items = sorted(_encode(v)["text"] for v in value)
            return {"kind": "set", "text": "{" + ", ".join(items) + "}", "items": items}
        return {"kind": "expression", "text": sympy.sstr(value)[:MAX_TEXT]}

    if isinstance(value, set | frozenset):
        items = sorted(_encode(v)["text"] for v in value)
        return {"kind": "set", "text": "{" + ", ".join(items) + "}", "items": items}
    if isinstance(value, list | tuple):
        items = [_encode(v)["text"] for v in value]
        return {"kind": "set", "text": "(" + ", ".join(items) + ")", "items": items}
    if value is None:
        raise ValueError("solve() returned None; it must return the answer")
    return {"kind": "text", "text": str(value)[:MAX_TEXT]}


def _load(code: str) -> Any:
    namespace: dict[str, Any] = {"__builtins__": _restricted_builtins(), "__name__": "solver"}
    exec(compile(code, "<solver_code>", "exec"), namespace)
    solve = namespace.get("solve")
    if not callable(solve):
        raise ValueError("solver_code must define a function named solve()")
    return solve


def main() -> int:
    request = json.loads(sys.stdin.readline() or "{}")
    _apply_limits(int(request.get("memory_mb", 1024)), int(request.get("cpu_seconds", 60)))

    # User code must not be able to corrupt the protocol: keep a private handle
    # on the pipe, then point fd 1 at /dev/null so a stray print() is discarded
    # rather than parsed as a response.
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout = sys.stderr

    def reply(payload: dict[str, Any]) -> None:
        out.write(json.dumps(payload) + "\n")
        out.flush()

    try:
        solve = _load(request["code"])
    except BaseException as e:
        reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    reply({"ok": True, "ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            params = {k: _decode(v) for k, v in json.loads(line)["params"].items()}
            reply({"ok": True, "answer": _encode(solve(**params))})
        except BaseException as e:
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
