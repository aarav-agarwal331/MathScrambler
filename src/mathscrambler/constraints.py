"""Blueprint constraints: Python expressions over parameter slots.

A blueprint carries constraints like ``p1 % p2 == 0`` or ``triangle(p4, p5, p6)``
and the sampler evaluates them thousands of times per variant, so they run in
this process rather than the sandbox. That is only safe because every expression
is checked against an AST allowlist first — no attributes, no subscripts, no
calls except the named helpers below — and rejected outright if it steps outside.

The spec's own example writes one constraint in prose ("triangle inequality on
p4,p5,p6"). Prose is not evaluable, so instead of guessing at it the helpers
below give the model a vocabulary for the usual conditions, the extraction
prompt names them, and an expression that still will not compile is fed back as
a blueprint regeneration reason (Section 3, Step B) rather than silently dropped.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# A constant exponent ceiling. `p1 ** 2` is ordinary; `2 ** p1` with an unbounded
# slot would hang or exhaust memory here in the parent process, where there is no
# sandbox to catch it.
MAX_EXPONENT = 8
MAX_FACTORIAL = 20


class ConstraintError(ValueError):
    """A constraint could not be compiled or evaluated. The message names the cause."""


def _triangle(a: float, b: float, c: float) -> bool:
    """Three positive lengths that form a non-degenerate triangle."""
    return a > 0 and b > 0 and c > 0 and a + b > c and a + c > b and b + c > a


def _right_triangle(a: float, b: float, c: float) -> bool:
    """Legs `a`, `b` and hypotenuse `c`, exactly."""
    return _triangle(a, b, c) and math.isclose(a * a + b * b, c * c, rel_tol=1e-9)


def _distinct(*values: Any) -> bool:
    return len(set(values)) == len(values)


def _factorial(n: int) -> int:
    if not isinstance(n, int) or n < 0 or n > MAX_FACTORIAL:
        raise ConstraintError(f"factorial() takes an integer 0..{MAX_FACTORIAL}, got {n!r}")
    return math.factorial(n)


def _is_prime(n: Any) -> bool:
    if not isinstance(n, int) or n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % d != 0 for d in range(3, math.isqrt(n) + 1, 2))


def _is_square(n: Any) -> bool:
    return isinstance(n, int) and n >= 0 and math.isqrt(n) ** 2 == n


def _divides(a: Any, b: Any) -> bool:
    """True when `a` divides `b` evenly. `divides(0, x)` is False, never an error."""
    return a != 0 and b % a == 0


HELPERS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "between": lambda x, lo, hi: lo <= x <= hi,
    "ceil": math.ceil,
    "coprime": lambda a, b: math.gcd(int(a), int(b)) == 1,
    "distinct": _distinct,
    "divides": _divides,
    "factorial": _factorial,
    "floor": math.floor,
    "gcd": math.gcd,
    "int": int,
    "is_prime": _is_prime,
    "is_square": _is_square,
    "isqrt": math.isqrt,
    "lcm": math.lcm,
    "len": len,
    "max": max,
    "min": min,
    "right_triangle": _right_triangle,
    "round": round,
    "same_sign": lambda a, b: (a > 0) == (b > 0) and a != 0 and b != 0,
    "sum": sum,
    "triangle": _triangle,
}

HELP_TEXT = ", ".join(sorted(HELPERS))

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.IfExp,
    ast.GeneratorExp,
    ast.ListComp,
    ast.comprehension,
    ast.Store,
)


@dataclass(frozen=True)
class Constraint:
    """One compiled, AST-checked constraint. `source` is kept for error messages."""

    source: str
    code: Any  # a code object from compile()
    slots: frozenset[str]

    def evaluate(self, values: Mapping[str, Any]) -> bool:
        missing = self.slots - values.keys()
        if missing:
            raise ConstraintError(f"{self.source!r}: no value for {', '.join(sorted(missing))}")
        env = {**HELPERS, **{name: values[name] for name in self.slots}}
        try:
            result = eval(self.code, {"__builtins__": {}}, env)
        except ConstraintError:
            raise
        except Exception as e:
            raise ConstraintError(f"{self.source!r}: {type(e).__name__}: {e}") from e
        if not isinstance(result, bool | int | float):
            raise ConstraintError(
                f"{self.source!r}: must be a true/false condition, got {type(result).__name__}"
            )
        return bool(result)


def _bound_names(tree: ast.AST) -> set[str]:
    """Names a comprehension binds locally — `x` in `all(x > 0 for x in ...)`."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    return bound


def compile_constraint(source: str, slots: Iterable[str]) -> Constraint:
    """Check `source` against the allowlist and compile it. Raises ConstraintError."""
    known = set(slots)
    text = source.strip()
    if not text:
        raise ConstraintError("empty constraint")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ConstraintError(
            f"{text!r} is not a Python expression ({e.msg}). Constraints must be expressions over the "
            f"parameter slots; available helpers: {HELP_TEXT}"
        ) from e

    local = _bound_names(tree)
    used: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ConstraintError(f"{text!r}: {type(node).__name__} is not allowed in a constraint")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in HELPERS:
                name = getattr(node.func, "id", ast.dump(node.func))
                raise ConstraintError(f"{text!r}: cannot call {name}; available helpers: {HELP_TEXT}")
            if node.keywords:
                raise ConstraintError(f"{text!r}: keyword arguments are not allowed")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = node.right
            if not (
                isinstance(exponent, ast.Constant)
                and isinstance(exponent.value, int)
                and 0 <= exponent.value <= MAX_EXPONENT
            ):
                raise ConstraintError(
                    f"{text!r}: ** needs a literal exponent between 0 and {MAX_EXPONENT} "
                    "(a slot-sized exponent could be astronomically large)"
                )
        elif isinstance(node, ast.Constant) and not isinstance(node.value, int | float | None):
            # Numbers only. A string constant plus the multiplication operator
            # ("a" * 10**8) would allocate hundreds of megabytes in *this*
            # process, and no constraint over numeric slots needs one.
            raise ConstraintError(f"{text!r}: only numeric literals are allowed, not {node.value!r}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in HELPERS:
            if node.id not in known and node.id not in local:
                raise ConstraintError(
                    f"{text!r}: unknown name {node.id!r}; parameter slots are "
                    f"{', '.join(sorted(known)) or '(none)'}"
                )
            if node.id in known:
                used.add(node.id)

    return Constraint(source=text, code=compile(tree, "<constraint>", "eval"), slots=frozenset(used))


def compile_all(sources: Iterable[str], slots: Iterable[str]) -> tuple[list[Constraint], list[str]]:
    """Compile every constraint; return the good ones and the errors, never raising.

    Blueprint validation wants *all* the problems at once so one regeneration
    round can fix them together, rather than one per round.
    """
    known = list(slots)
    compiled: list[Constraint] = []
    errors: list[str] = []
    for source in sources:
        try:
            compiled.append(compile_constraint(source, known))
        except ConstraintError as e:
            errors.append(str(e))
    return compiled, errors


def satisfied(constraints: Iterable[Constraint], values: Mapping[str, Any]) -> bool:
    """True when every constraint holds. An evaluation error means "no"."""
    try:
        return all(c.evaluate(values) for c in constraints)
    except ConstraintError:
        return False


def failures(constraints: Iterable[Constraint], values: Mapping[str, Any]) -> list[str]:
    """Which constraints do not hold for `values` — for reporting, not sampling."""
    out: list[str] = []
    for constraint in constraints:
        try:
            if not constraint.evaluate(values):
                out.append(constraint.source)
        except ConstraintError as e:
            out.append(str(e))
    return out
