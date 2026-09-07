"""Variant sampling: new numbers, same problem (Section 3, Step C).

Deterministic and model-free. Given a blueprint and a seed this draws candidate
parameters, throws away every draw that breaks a constraint, produces an answer
of the wrong shape, or lands on a degenerate answer, and keeps what survives.
The same seed draws the same variants, which is what makes a run reproducible.

Rejection sampling is the whole point: asking a model for "the same problem with
different numbers" is what produces non-integer answers and impossible triangles.
Here the numbers are drawn under the blueprint's own constraints and the answer
is computed by the blueprint's own solver, never guessed.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from mathscrambler.blueprint import Blueprint, compiled_constraints, render_template
from mathscrambler.constraints import Constraint, satisfied
from mathscrambler.sandbox import Answer, SandboxError, SolveResult, SolverSession

Number = int | float | Fraction

# A solver that times out kills its session. Restarting is worth doing a few
# times — one pathological draw should not end the run — but a solver that keeps
# hanging is a broken blueprint, not bad luck.
MAX_SOLVER_RESTARTS = 3

# Nouns that carry mathematical meaning: swapping them changes the problem
# rather than the scenery. "square ABCD" may become "square PQRS", never
# "rectangle PQRS" (Section 3, Step C: "keep a guard list").
GUARDED_NOUNS = frozenset(
    {
        "angle", "area", "average", "bisector", "circle", "circumference", "composite",
        "cube", "denominator", "diagonal", "diameter", "difference", "digit", "divisor",
        "even", "factor", "fraction", "hypotenuse", "integer", "isosceles", "mean",
        "median", "midpoint", "mode", "multiple", "numerator", "odd", "parallel",
        "percent", "percentage", "perimeter", "perpendicular", "prime", "product",
        "quotient", "radius", "range", "ratio", "rectangle", "remainder", "root",
        "sequence", "series", "square", "sum", "tangent", "triangle", "vertex",
        "volume", "whole",
    }
)

_WORD_RE = re.compile(r"[a-z]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_MATH_RE = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")


# --------------------------------------------------------------------------- draws


def _int_range(value: int, *, relaxed: bool) -> tuple[int, int]:
    """The magnitude class of `value`: same digit count, or 4x either way if relaxed.

    Magnitude matters because "17 apples" and "4,000,000 apples" are not the same
    problem to a reader even when they are the same problem to the solver.
    """
    magnitude = abs(value)
    if magnitude < 10:
        low, high = 1, 9
    else:
        digits = len(str(magnitude))
        low, high = 10 ** (digits - 1), 10**digits - 1
    if relaxed:
        # A union with the strict class, never a replacement for it: `21` has a
        # digit class of 10..99 but a 4x band of 5..84, and relaxing must not
        # take away draws that were allowed before.
        return min(low, max(1, magnitude // 4)), max(high, magnitude * 4)
    return low, high


def _draw_int(rng: random.Random, value: int, *, relaxed: bool) -> int:
    low, high = _int_range(value, relaxed=relaxed)
    drawn = rng.randint(low, high)
    return -drawn if value < 0 else drawn


def _draw_float(rng: random.Random, value: float, *, relaxed: bool) -> float:
    places = len(str(value).partition(".")[2])
    span = 4.0 if relaxed else 2.0
    magnitude = abs(value) or 1.0
    drawn = rng.uniform(magnitude / span, magnitude * span)
    return round(-drawn if value < 0 else drawn, max(places, 1))


def _draw_fraction(rng: random.Random, value: Fraction, *, relaxed: bool) -> Fraction:
    numerator = _draw_int(rng, value.numerator or 1, relaxed=relaxed)
    denominator = _draw_int(rng, value.denominator, relaxed=relaxed) or 1
    return Fraction(numerator, denominator)


def draw_parameters(bp: Blueprint, rng: random.Random, *, relaxed: bool = False) -> dict[str, Number]:
    """One candidate draw. Says nothing about whether it is any good."""
    drawn: dict[str, Number] = {}
    for param in bp.parameters:
        value = param.value
        if isinstance(value, Fraction):
            drawn[param.slot] = _draw_fraction(rng, value, relaxed=relaxed)
        elif isinstance(value, int):
            drawn[param.slot] = _draw_int(rng, value, relaxed=relaxed)
        else:
            drawn[param.slot] = _draw_float(rng, value, relaxed=relaxed)
    return drawn


# --------------------------------------------------------------------------- answer shape


def _terminates(answer: Answer) -> bool:
    """Does this rational have a terminating decimal expansion?"""
    if answer.kind == "integer":
        return True
    _, _, denominator = answer.text.partition("/")
    if not denominator.isdigit():
        return False
    remaining = int(denominator)
    for prime in (2, 5):
        while remaining % prime == 0:
            remaining //= prime
    return remaining == 1


def answer_is_nice_like(original: Answer, candidate: Answer) -> bool:
    """Is the candidate's answer the same *kind* of answer as the original's?

    Integer stays integer, terminating stays terminating (Section 3, Step C).
    A variant whose answer turns out to be 47/93 when the original's was 8 has
    the same solution path but is not the same exercise.
    """
    if original.kind == "integer":
        return candidate.kind == "integer"
    if original.kind == "rational":
        if candidate.kind not in ("integer", "rational"):
            return False
        return _terminates(candidate) if _terminates(original) else True
    if original.kind == "float":
        return candidate.kind in ("integer", "rational", "float")
    return candidate.kind == original.kind


def degeneracy(answer: Answer, params: Mapping[str, Number]) -> set[str]:
    """Ways in which this answer is uninformative: 0, 1, -1, or an input echoed back."""
    reasons: set[str] = set()
    if answer.numeric is not None:
        if answer.numeric == 0:
            reasons.add("zero")
        elif answer.numeric == 1:
            reasons.add("one")
        elif answer.numeric == -1:
            reasons.add("minus one")
    if any(answer.text == str(value) for value in params.values()):
        reasons.add("equal to an input")
    return reasons


def _sign(value: float | None) -> int:
    return 0 if value is None else (value > 0) - (value < 0)


# --------------------------------------------------------------------------- entities


def check_entity_swap(original: str, replacement: str) -> str | None:
    """Why this swap is not allowed, or None if it is.

    A replacement must keep every mathematically loaded word the original had:
    "square ABCD" -> "square PQRS" is scenery, "square ABCD" -> "rectangle PQRS"
    is a different problem.
    """
    if not replacement.strip():
        return "replacement is empty"
    if replacement.strip().lower() == original.strip().lower():
        return "replacement is the same as the original"
    kept = set(_WORD_RE.findall(replacement.lower()))
    for word in _WORD_RE.findall(original.lower()):
        if word in GUARDED_NOUNS and word not in kept:
            return f"{word!r} carries mathematical meaning and must survive the swap"
    if _NUMBER_RE.search(replacement):
        return "entity replacements must not introduce numbers"
    return None


def math_and_numbers_unchanged(before: str, after: str) -> bool:
    """Gate for the one allowed grammar pass: articles may move, maths may not.

    The `fast` model is allowed to fix "a apples" into "some apples". It is not
    allowed to touch a number or anything inside math delimiters, and the only
    way to be sure of that is to compare them (Section 3, Step C).
    """
    return _NUMBER_RE.findall(before) == _NUMBER_RE.findall(after) and _MATH_RE.findall(
        before
    ) == _MATH_RE.findall(after)


# --------------------------------------------------------------------------- sampling


@dataclass(frozen=True)
class Variant:
    index: int
    params: dict[str, Number]
    entities: dict[str, str]
    answer: Answer
    statement_md: str
    attempts: int
    relaxed: bool


@dataclass
class SamplingReport:
    """What came out, and — when something did not — exactly why."""

    variants: list[Variant] = field(default_factory=list)
    attempts: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    failure: str | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None

    def summary(self) -> str:
        if not self.rejections:
            return f"{len(self.variants)} variants in {self.attempts} attempts"
        worst = ", ".join(f"{reason}: {n}" for reason, n in self.rejections.most_common(4))
        return f"{len(self.variants)} variants in {self.attempts} attempts ({worst})"


class _Solver:
    """A solver session that reopens itself after a timeout, a bounded number of times."""

    def __init__(self, code: str, *, timeout_s: float, memory_mb: int) -> None:
        self._code = code
        self._timeout_s = timeout_s
        self._memory_mb = memory_mb
        self._session: SolverSession | None = None
        self.restarts = 0

    def solve(self, params: Mapping[str, Number]) -> SolveResult:
        if self._session is None:
            if self.restarts > MAX_SOLVER_RESTARTS:
                return SolveResult(ok=False, error="solver kept failing to run; giving up")
            try:
                self._session = SolverSession(
                    self._code, timeout_s=self._timeout_s, memory_mb=self._memory_mb
                )
                self._session.start()
            except SandboxError as e:
                self._session = None
                self.restarts = MAX_SOLVER_RESTARTS + 1
                return SolveResult(ok=False, error=f"solver_code could not be loaded: {e}")
        result = self._session.solve(params)
        # A timeout or a dead child ends the session; ordinary solver exceptions
        # (a ValueError on a bad draw) leave it perfectly usable.
        if not result.ok and result.error and (result.timed_out or "died" in result.error):
            self.close()
            self.restarts += 1
        return result

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> _Solver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def sample_variants(
    bp: Blueprint,
    original_answer: Answer,
    *,
    n: int,
    seed: int = 0,
    max_samples: int = 2000,
    relaxed_extra_samples: int = 1000,
    keep_nice_answers: bool = True,
    reject_degenerate: bool = True,
    entity_maps: Sequence[Mapping[str, str]] | None = None,
    timeout_s: float = 5.0,
    memory_mb: int = 1024,
) -> SamplingReport:
    """`n` variants of `bp`, or a report saying why there are fewer.

    `entity_maps[i]` supplies variant i's entity replacements (the `fast` model
    proposes them upstream); without one the original entities are kept, so the
    numbers still change and the sampler stays usable with no model at all.
    """
    report = SamplingReport()
    if bp.solver_code is None:
        report.failure = "this blueprint has no solver (proof kind); sampling needs one"
        return report

    constraints = compiled_constraints(bp)
    original_params = bp.original_params()
    original_degeneracy = degeneracy(original_answer, original_params)
    original_sign = _sign(original_answer.numeric)
    seen: set[tuple[Any, ...]] = {tuple(str(v) for v in original_params.values())}

    with _Solver(bp.solver_code, timeout_s=timeout_s, memory_mb=memory_mb) as solver:
        for index in range(n):
            # Seeded per variant, not per run: a rejected draw for variant 1 must
            # not shift what variant 2 draws, or adding a constraint would silently
            # change every later variant.
            rng = random.Random(f"{seed}:{index}")
            variant = _sample_one(
                bp,
                rng,
                index=index,
                solver=solver,
                constraints=constraints,
                original_answer=original_answer,
                original_degeneracy=original_degeneracy,
                original_sign=original_sign,
                seen=seen,
                report=report,
                max_samples=max_samples,
                relaxed_extra_samples=relaxed_extra_samples,
                keep_nice_answers=keep_nice_answers,
                reject_degenerate=reject_degenerate,
                entities=dict(entity_maps[index]) if entity_maps and index < len(entity_maps) else None,
            )
            if variant is None:
                report.failure = (
                    f"could not find variant {index + 1} of {n} after "
                    f"{max_samples + relaxed_extra_samples} attempts ({report.summary()})"
                )
                return report
            report.variants.append(variant)
    return report


def _sample_one(
    bp: Blueprint,
    rng: random.Random,
    *,
    index: int,
    solver: _Solver,
    constraints: list[Constraint],
    original_answer: Answer,
    original_degeneracy: set[str],
    original_sign: int,
    seen: set[tuple[Any, ...]],
    report: SamplingReport,
    max_samples: int,
    relaxed_extra_samples: int,
    keep_nice_answers: bool,
    reject_degenerate: bool,
    entities: dict[str, str] | None,
) -> Variant | None:
    originals = bp.original_params()
    entity_values = entities or bp.original_entities()

    for attempt in range(1, max_samples + relaxed_extra_samples + 1):
        relaxed = attempt > max_samples
        report.attempts += 1
        params = draw_parameters(bp, rng, relaxed=relaxed)

        # Criterion: every number changes. A variant sharing a number with its
        # original reads as a typo of it rather than a new problem.
        if any(params[slot] == originals[slot] for slot in params):
            report.rejections["a parameter kept its original value"] += 1
            continue
        key = tuple(str(v) for v in params.values())
        if key in seen:
            report.rejections["duplicate draw"] += 1
            continue
        if not satisfied(constraints, params):
            report.rejections["constraints not satisfied"] += 1
            continue

        result = solver.solve(params)
        if not result.ok or result.answer is None:
            report.rejections[f"solver: {result.error}"] += 1
            if solver.restarts > MAX_SOLVER_RESTARTS:
                return None
            continue

        answer = result.answer
        if keep_nice_answers and not answer_is_nice_like(original_answer, answer):
            report.rejections[f"answer was {answer.kind}, original was {original_answer.kind}"] += 1
            continue
        if reject_degenerate:
            if new_reasons := degeneracy(answer, params) - original_degeneracy:
                report.rejections[f"degenerate answer ({', '.join(sorted(new_reasons))})"] += 1
                continue
            if original_sign and _sign(answer.numeric) == -original_sign:
                report.rejections["answer flipped sign"] += 1
                continue

        seen.add(key)
        return Variant(
            index=index,
            params=params,
            entities=entity_values,
            answer=answer,
            statement_md=render_template(bp.template_md, {**entity_values, **params}),
            attempts=attempt,
            relaxed=relaxed,
        )
    return None
