"""The blueprint: a problem's logical skeleton, with its numbers pulled out.

Section 3, Step B. The `reasoner` produces one of these per problem; everything
here is the deterministic half — the schema, the validators that decide whether
a blueprint is usable, and the gate that makes the solver prove itself on the
original numbers before any variant is sampled.

A blueprint that fails validation is not patched up: the errors are collected
and fed back for regeneration (<= 3 tries, Section 3). Guessing at what a model
meant is how a variant ends up with a different solution path than its original.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mathscrambler.constraints import HELPERS, Constraint, compile_all
from mathscrambler.constraints import failures as constraint_failures
from mathscrambler.sandbox import Answer, SandboxError, SolverSession

Kind = Literal["computational", "proof", "mixed"]
AnswerType = Literal["integer", "rational", "expression", "set", "proof"]
ParamType = Literal["int", "float", "fraction"]

ENTITY_SLOT_RE = re.compile(r"^E\d+$")
PARAM_SLOT_RE = re.compile(r"^p\d+$")
# Only `{E1}`/`{p2}`-shaped braces are placeholders. LaTeX is full of braces —
# `\sqrt{x}`, `\frac{1}{2}` — and a looser pattern would read them as slots.
PLACEHOLDER_RE = re.compile(r"\{([Ep]\d+)\}")


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: str
    original: str
    role: str


class Parameter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: str
    original: int | float | str
    type: ParamType = "int"
    description: str = ""
    constraints: str = ""

    @property
    def value(self) -> int | float | Fraction:
        """The original as a Python number. Raises ValueError if it is not one."""
        if self.type == "int":
            return int(self.original)
        if self.type == "float":
            return float(self.original)
        return Fraction(str(self.original))


class Blueprint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Kind
    domain: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    template_md: str
    solution_outline: list[str] = Field(default_factory=list)
    solver_code: str | None = None
    answer_type: AnswerType
    invariants: list[str] = Field(default_factory=list)

    @property
    def parameter_slots(self) -> list[str]:
        return [p.slot for p in self.parameters]

    @property
    def entity_slots(self) -> list[str]:
        return [e.slot for e in self.entities]

    def original_params(self) -> dict[str, int | float | Fraction]:
        return {p.slot: p.value for p in self.parameters}

    def original_entities(self) -> dict[str, str]:
        return {e.slot: e.original for e in self.entities}


# --------------------------------------------------------------------------- validation


def _solver_signature(code: str) -> list[str] | str:
    """Parameter names of `def solve(...)`, or a message saying what is wrong.

    Parsed, never executed: a blueprint whose solver has the wrong signature
    should be reported as such, not discovered as a TypeError two layers down.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"solver_code does not parse: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "solve":
            args = node.args
            if args.vararg or args.kwarg:
                return (
                    "solver_code: solve() must take one named argument per parameter "
                    "slot, not *args/**kwargs"
                )
            return [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return "solver_code defines no function named solve()"


def validate_blueprint(bp: Blueprint) -> list[str]:
    """Everything wrong with `bp`, in one list. Empty means usable.

    All the problems at once, deliberately: one regeneration round should be able
    to fix everything, instead of surfacing the next fault only after the last.
    """
    problems: list[str] = []

    slots_seen: set[str] = set()
    for entity in bp.entities:
        if not ENTITY_SLOT_RE.match(entity.slot):
            problems.append(f"entity slot {entity.slot!r} must look like E1, E2, ...")
        if not entity.original.strip():
            problems.append(f"entity {entity.slot}: `original` is empty")
        if entity.slot in slots_seen:
            problems.append(f"duplicate slot {entity.slot!r}")
        slots_seen.add(entity.slot)

    for param in bp.parameters:
        if not PARAM_SLOT_RE.match(param.slot):
            problems.append(f"parameter slot {param.slot!r} must look like p1, p2, ...")
        if param.slot in slots_seen:
            problems.append(f"duplicate slot {param.slot!r}")
        if param.slot in HELPERS:
            problems.append(f"parameter slot {param.slot!r} shadows a constraint helper of the same name")
        slots_seen.add(param.slot)
        try:
            _ = param.value
        except (TypeError, ValueError, ZeroDivisionError):
            problems.append(
                f"parameter {param.slot}: original {param.original!r} is not a valid {param.type}"
            )

    used = set(PLACEHOLDER_RE.findall(bp.template_md))
    for slot in (*bp.entity_slots, *bp.parameter_slots):
        if slot not in used:
            problems.append(f"template_md never uses {{{slot}}}, so varying it would change nothing")
    for slot in sorted(used - slots_seen):
        problems.append(f"template_md uses {{{slot}}}, which is not a declared slot")

    _, constraint_errors = compile_all(bp.constraints, bp.parameter_slots)
    problems.extend(constraint_errors)

    if not bp.solution_outline:
        problems.append("solution_outline is empty; the variant has nothing to be checked against")

    is_proof = bp.answer_type == "proof"
    if is_proof != (bp.solver_code is None):
        problems.append(
            'answer_type "proof" and a null solver_code must go together '
            f"(answer_type={bp.answer_type!r}, solver_code={'null' if bp.solver_code is None else 'present'})"
        )
    if bp.kind == "proof" and not is_proof:
        problems.append('kind "proof" needs answer_type "proof"')

    if bp.solver_code is not None:
        signature = _solver_signature(bp.solver_code)
        if isinstance(signature, str):
            problems.append(signature)
        else:
            expected = set(bp.parameter_slots)
            missing = expected - set(signature)
            extra = set(signature) - expected
            if missing:
                problems.append(f"solve() is missing arguments for {', '.join(sorted(missing))}")
            if extra:
                problems.append(f"solve() takes {', '.join(sorted(extra))}, which are not parameter slots")

    return problems


# --------------------------------------------------------------------------- solver gate


@dataclass(frozen=True)
class SolverCheck:
    """Did the solver reproduce the original problem? `ok` gates variant sampling."""

    ok: bool
    answer: Answer | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def feedback(self) -> str:
        return "; ".join(self.problems)


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*(?:/\d+)?")


def answer_matches_given(answer: Answer, given: str) -> bool:
    """Is `answer` consistent with the answer printed in the source?

    Deliberately lenient. `given_answer` is whatever the source said — "$14",
    "14 pens", "x = 8" — so this asks whether the computed value appears in it,
    not whether the strings match. A stricter comparison here would reject
    correct solvers over a currency symbol; Step D is where exactness lives.
    """
    text = given.strip()
    if not text:
        return False
    if answer.numeric is None:
        # Expressions and sets have no value to compare, so a substring match is
        # all there is. Numbers must NOT take this path: "4" is a substring of
        # "$41", and a solver returning 4 would look like it reproduced 41.
        return answer.text.lower() in text.lower()
    for token in _NUMBER_RE.findall(text.replace(",", "")):
        try:
            value = float(Fraction(token)) if "/" in token else float(token)
        except (ValueError, ZeroDivisionError):
            continue
        if abs(value - answer.numeric) <= 1e-9 * max(1.0, abs(value)):
            return True
    return False


def check_solver(
    bp: Blueprint,
    *,
    given_answer: str | None = None,
    timeout_s: float = 5.0,
    memory_mb: int = 1024,
) -> SolverCheck:
    """Run the solver on the original parameters and judge the result.

    Three ways to fail, all of them reasons to regenerate the blueprint rather
    than to sample from it: the blueprint's own constraints reject its original
    numbers, the solver errors or times out, or it computes something other than
    the answer the source printed.
    """
    if bp.solver_code is None:
        return SolverCheck(ok=True)  # a proof blueprint has nothing to run

    problems: list[str] = []
    compiled, errors = compile_all(bp.constraints, bp.parameter_slots)
    problems.extend(errors)

    try:
        params = bp.original_params()
    except (TypeError, ValueError, ZeroDivisionError) as e:
        return SolverCheck(ok=False, problems=[*problems, f"original parameters are not numbers: {e}"])

    if unmet := constraint_failures(compiled, params):
        problems.append(
            "the original parameters do not satisfy the blueprint's own constraints: " + "; ".join(unmet)
        )

    try:
        with SolverSession(bp.solver_code, timeout_s=timeout_s, memory_mb=memory_mb) as session:
            result = session.solve(params)
    except SandboxError as e:
        return SolverCheck(ok=False, problems=[*problems, f"solver_code could not be loaded: {e}"])

    if not result.ok or result.answer is None:
        return SolverCheck(
            ok=False, problems=[*problems, f"solve() failed on the original values: {result.error}"]
        )

    answer = result.answer
    if given_answer and not answer_matches_given(answer, given_answer):
        problems.append(
            f"solve() returned {answer.text!r} but the source's answer is {given_answer!r}; "
            "the solver does not reproduce the original problem"
        )
    return SolverCheck(ok=not problems, answer=answer, problems=problems)


def compiled_constraints(bp: Blueprint) -> list[Constraint]:
    """Compile the constraints, assuming validation already passed."""
    compiled, errors = compile_all(bp.constraints, bp.parameter_slots)
    if errors:
        raise ValueError("blueprint has invalid constraints: " + "; ".join(errors))
    return compiled


def render_template(template: str, values: dict[str, Any]) -> str:
    """Substitute `{E1}`/`{p2}` placeholders, leaving every other brace alone.

    `str.format` cannot be used here: a statement containing `\\frac{1}{2}` would
    make it raise, or worse, silently eat the braces.
    """

    def replace(match: re.Match[str]) -> str:
        slot = match.group(1)
        return str(values[slot]) if slot in values else match.group(0)

    return PLACEHOLDER_RE.sub(replace, template)
