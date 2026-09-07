from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import pytest

from mathscrambler.sandbox import SandboxError, SolverSession, run_solver

REPO = Path(__file__).resolve().parent.parent


def test_runs_a_solver_and_types_the_answer():
    result = run_solver("def solve(a, b):\n    return a * b + 1\n", {"a": 6, "b": 7})
    assert result.ok
    assert result.answer is not None
    assert (result.answer.kind, result.answer.text, result.answer.numeric) == ("integer", "43", 43.0)


def test_fractions_survive_the_round_trip_and_collapse_when_whole():
    code = "def solve(n, d):\n    from fractions import Fraction\n    return Fraction(n, d)\n"
    assert run_solver(code, {"n": 3, "d": 4}).answer.text == "3/4"
    whole = run_solver(code, {"n": 6, "d": 2}).answer
    # Whether a solver returns Fraction(6, 2) or 6 is an implementation detail;
    # if it leaked out, the same answer twice would compare unequal in Step D.
    assert (whole.kind, whole.text) == ("integer", "3")
    passed_in = run_solver("def solve(f):\n    return f * 2\n", {"f": Fraction(3, 8)}).answer
    assert passed_in.text == "3/4"


def test_a_session_serves_many_solves_without_respawning():
    with SolverSession("def solve(n):\n    return n * n\n") as session:
        started = time.monotonic()
        answers = [session.solve({"n": i}).answer.text for i in range(50)]
        elapsed = time.monotonic() - started
    assert answers[:4] == ["0", "1", "4", "9"]
    # The whole point of keeping the child alive: rejection sampling makes
    # hundreds of these per variant, and a fresh interpreter each time would
    # cost tens of milliseconds apiece.
    assert elapsed < 1.0, f"50 solves took {elapsed:.2f}s; the session is being respawned"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("import socket\n    return 1", "may not import 'socket'"),
        ("import os\n    return 1", "may not import 'os'"),
        ("return open('/etc/passwd').read()", "open"),
        ("return eval('1+1')", "eval"),
        # `__import__` has to stay in builtins for `import` to work at all;
        # it is the guarded one, so reaching for it directly changes nothing.
        ("return __import__('os').getcwd()", "may not import 'os'"),
    ],
)
def test_denied_imports_and_builtins(body: str, expected: str):
    assert expected in run_solver(f"def solve():\n    {body}\n", {}).error


def test_a_solver_cannot_write_to_the_repository():
    """Inputs and the repo are never touched by model-written code (Section 1.2)."""
    target = REPO / "pyproject.toml"
    before = target.read_bytes()
    result = run_solver(f"def solve():\n    open({str(target)!r}, 'w').write('x')\n    return 1\n", {})
    assert not result.ok
    assert target.read_bytes() == before


def test_timeout_is_reported_and_ends_the_session():
    with SolverSession("def solve():\n    while True:\n        pass\n", timeout_s=1.0) as session:
        started = time.monotonic()
        result = session.solve({})
        assert result.timed_out and "timeout" in result.error
        assert time.monotonic() - started < 3.0
        # The child was killed mid-line; a later solve must fail cleanly rather
        # than read a stale or half-written response.
        assert not session.solve({}).ok


def test_a_solver_exception_is_data_and_the_session_survives_it():
    with SolverSession("def solve(n):\n    return 10 // n\n") as session:
        failed = session.solve({"n": 0})
        assert not failed.ok and "ZeroDivisionError" in failed.error
        assert session.solve({"n": 5}).answer.text == "2"


def test_stray_output_cannot_corrupt_the_protocol():
    # `sys` is not even importable, so a solver cannot reach the real stdout;
    # what it can do is print, and a printed response-shaped line must not be
    # mistaken for a response.
    forged = '{"ok": true, "answer": {"kind": "integer", "text": "0"}}'
    code = f"def solve(n):\n    print({forged!r})\n    return n\n"
    with SolverSession(code) as session:
        assert session.solve({"n": 7}).answer.text == "7"
        assert session.solve({"n": 8}).answer.text == "8"


def test_load_failures_raise_with_the_reason():
    with pytest.raises(SandboxError, match="SyntaxError"):
        SolverSession("def solve(:\n").start()
    with pytest.raises(SandboxError, match="must define a function named solve"):
        SolverSession("x = 1\n").start()
    with pytest.raises(SandboxError, match="may not import"):
        SolverSession("import socket\ndef solve():\n    return 1\n").start()


def test_run_solver_reports_a_load_failure_instead_of_raising():
    result = run_solver("def solve(:\n", {})
    assert not result.ok and "SyntaxError" in result.error


def test_sympy_answers_are_classified():
    code = (
        "def solve(n):\n    import sympy\n    x = sympy.Symbol('x')\n"
        "    return sympy.simplify(x * n + x)\n"
    )
    answer = run_solver(code, {"n": 2}).answer
    assert answer.kind == "expression" and "x" in answer.text
    roots = run_solver(
        "def solve(n):\n    import sympy\n    x = sympy.Symbol('x')\n"
        "    return set(sympy.solve(x**2 - n, x))\n",
        {"n": 9},
    ).answer
    assert roots.kind == "set" and sorted(roots.items) == ["-3", "3"]


def test_returning_nothing_is_an_error_not_a_null_answer():
    result = run_solver("def solve():\n    return None\n", {})
    assert not result.ok and "must return the answer" in result.error


def test_unserializable_parameters_are_refused_before_the_child_sees_them():
    result = run_solver("def solve(x):\n    return 1\n", {"x": object()})
    assert not result.ok and "parameter values must be" in result.error


def test_a_child_that_cannot_start_says_so_instead_of_timing_out(monkeypatch):
    """A 20-second "did not load" for a child that died instantly would send
    whoever reads it looking in entirely the wrong place."""
    import mathscrambler.sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod, "CHILD", Path("/nonexistent/_solver_child.py"))
    with pytest.raises(SandboxError, match="exited before loading"):
        SolverSession("def solve():\n    return 1\n").start()
