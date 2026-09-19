"""Blueprint extraction: one `Problem` in, one validated `Blueprint` out (Step B).

The `reasoner` writes the blueprint; nothing here trusts it. Every reply goes
through `validate_blueprint` (shape, slots, constraints) and then
`check_solver` (the solver must run on the original numbers and reproduce the
source's answer). A reply that fails either is sent back with the full list of
faults, at most `max_retries` times — the model fixes its own blueprint, the
code never patches one up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mathscrambler import prompts
from mathscrambler.blueprint import Blueprint, check_solver, validate_blueprint
from mathscrambler.ingest.models import Problem
from mathscrambler.ollama_client import OllamaClient, StructuredCallError, TimingRecord
from mathscrambler.sandbox import Answer


@dataclass(frozen=True)
class ExtractionResult:
    """A usable blueprint, or every reason the last attempt was not one."""

    blueprint: Blueprint | None
    answer: Answer | None  # the solver's answer on the original values; None for proofs
    attempts: int
    problems: list[str] = field(default_factory=list)
    timings: list[TimingRecord] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)  # for run.log

    @property
    def ok(self) -> bool:
        return self.blueprint is not None


def request_schema(*, proof: bool = False) -> dict[str, Any]:
    """The Blueprint schema with nothing optional.

    Pydantic marks fields that have defaults as not required, and under
    schema-constrained decoding the reasoner takes that literally: it reasons
    at length, then emits a JSON object that skips `entities`, `parameters`,
    `solution_outline` and `solver_code` altogether (observed live, every
    time). Requiring every property, and at least one parameter and outline
    step, makes the grammar itself demand the blueprint the prompt asked for.

    A declared proof may legitimately have no parameters (an induction over
    all n has no structural constant to vary), so `proof=True` lifts that one
    minimum — forcing it produced an empty parameter (observed live too).
    """
    schema = Blueprint.model_json_schema()

    def require_all(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                node["required"] = list(node["properties"])
            for value in node.values():
                require_all(value)
        elif isinstance(node, list):
            for value in node:
                require_all(value)

    require_all(schema)
    for key in ("solution_outline",) if proof else ("parameters", "solution_outline"):
        schema["properties"][key]["minItems"] = 1
    return schema


def _user_prompt(problem: Problem) -> str:
    return prompts.render(
        "blueprint_extract_user",
        statement_md=problem.statement_md,
        given_answer=problem.given_answer,
        diagram_description=problem.diagram_description,
        kind_hint=problem.kind_hint,
    )


async def extract_blueprint(
    client: OllamaClient,
    problem: Problem,
    *,
    max_retries: int = 3,
    timeout_s: float = 5.0,
    memory_mb: int = 1024,
) -> ExtractionResult:
    """Ask the reasoner for a blueprint until one passes both gates, or give up.

    `max_retries` counts blueprints judged, not HTTP calls: the client already
    retries schema-invalid JSON internally. A blueprint that never validates as
    JSON at all still burns an attempt — the model gets a fresh try, not the
    same conversation, because there is no reply to correct.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.load("blueprint_extract")},
        {"role": "user", "content": _user_prompt(problem)},
    ]
    # What goes to run.log: the conversation, plus notes on replies the client
    # itself rejected (never sent back to the model — Ollama knows no such role).
    transcript: list[dict[str, Any]] = list(messages)
    timings: list[TimingRecord] = []
    last_problems: list[str] = []
    attempts = 0
    for attempts in range(1, max_retries + 1):
        try:
            schema = request_schema(proof=problem.kind_hint == "proof")
            result = await client.structured("reasoner", messages, Blueprint, schema=schema)
        except StructuredCallError as e:
            last_problems = [str(e)]
            transcript.append({"role": "client", "content": str(e)})
            continue
        timings.extend(result.timings)
        for why in result.errors:
            transcript.append({"role": "client", "content": f"rejected an earlier reply: {why}"})
        bp = result.value
        problems = validate_blueprint(bp)
        if bp.solver_code is not None and not bp.parameters:
            problems.append("a computational blueprint needs at least one parameter to vary")
        answer: Answer | None = None
        if not problems:
            check = check_solver(
                bp, given_answer=problem.given_answer, timeout_s=timeout_s, memory_mb=memory_mb
            )
            problems = check.problems
            answer = check.answer
        reply = {"role": "assistant", "content": bp.model_dump_json(indent=2)}
        messages = [*messages, reply]
        transcript.append(reply)
        if not problems:
            return ExtractionResult(bp, answer, attempts, [], timings, transcript)
        last_problems = problems
        feedback = {"role": "user", "content": prompts.render("blueprint_feedback", problems=problems)}
        messages = [*messages, feedback]
        transcript.append(feedback)
    return ExtractionResult(None, None, attempts, last_problems, timings, transcript)
