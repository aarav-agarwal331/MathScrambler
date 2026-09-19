"""Results on disk: `results.json` (everything) and `results.md` (readable) — Step E.

The JSON is the record: original problem, blueprint, the solver's reference
answer, every variant with its parameter and entity maps, the sampling
statistics, and what failed and why. The Markdown is the same content laid out
for a person, answers folded under `<details>` so a page of variants can be
handed out as-is.
"""

from __future__ import annotations

import json
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mathscrambler.blueprint import Blueprint
from mathscrambler.ingest.models import Problem
from mathscrambler.sampler import Number, Variant
from mathscrambler.sandbox import Answer

ProblemStatus = Literal["ok", "partial", "blueprint_failed", "sampling_failed", "proof_unsupported"]

JsonNumber = int | float | str


def json_number(value: Number) -> JsonNumber:
    """Fractions travel as "n/d" text; ints and floats as themselves."""
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    return value


class ValueChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    original: JsonNumber | str
    variant: JsonNumber | str


class VariantRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int  # 1-based, as a reader counts them
    statement_md: str
    answer: Answer
    parameter_map: dict[str, ValueChange]
    entity_map: dict[str, ValueChange]
    attempts: int  # draws it took the sampler to find this one
    relaxed: bool  # found only after relaxing the magnitude rules

    @classmethod
    def from_variant(cls, bp: Blueprint, variant: Variant) -> VariantRecord:
        originals = bp.original_params()
        entities = bp.original_entities()
        return cls(
            index=variant.index + 1,
            statement_md=variant.statement_md,
            answer=variant.answer,
            parameter_map={
                slot: ValueChange(original=json_number(originals[slot]), variant=json_number(value))
                for slot, value in variant.params.items()
            },
            entity_map={
                slot: ValueChange(original=entities[slot], variant=value)
                for slot, value in variant.entities.items()
                if slot in entities
            },
            attempts=variant.attempts,
            relaxed=variant.relaxed,
        )


class SamplingRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attempts: int
    rejections: dict[str, int] = Field(default_factory=dict)
    failure: str | None = None


class ProblemRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    original: Problem
    status: ProblemStatus
    error: str | None = None
    blueprint: Blueprint | None = None
    blueprint_attempts: int = 0
    original_answer: Answer | None = None  # computed by the solver on the original values
    entities_accepted: int | None = None  # distinct scenery proposals that passed the guard
    sampling: SamplingRecord | None = None
    variants: list[VariantRecord] = Field(default_factory=list)
    timings_s: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SkippedRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    reason: str


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    started: datetime
    finished: datetime
    wall_s: float
    seed: int
    profile: str
    models: dict[str, str]  # role -> tag
    server: str  # e.g. "private http://127.0.0.1:11435"
    inputs: list[str]
    variants_requested: int
    problems_total: int
    problems_ok: int
    variants_total: int
    avg_s_per_variant: float | None
    # Honesty marker: Step D (independent re-solve of each variant) is not built.
    verification: str = "not implemented (Step D): answers come from the blueprint's solver"


class Results(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run: RunRecord
    problems: list[ProblemRecord]
    skipped: list[SkippedRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- markdown


def _details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary>\n\n{body}\n\n</details>"


def _answer_text(answer: Answer | None) -> str:
    return "unknown" if answer is None else f"`{answer.text}`"


def _problem_md(record: ProblemRecord) -> list[str]:
    original = record.original
    lines = [f"## {original.id}", "", f"*Source: {original.source} (problem {original.source_index})*", ""]
    lines += ["**Original**", "", original.statement_md, ""]

    if record.blueprint is not None:
        bp = record.blueprint
        body = [f"Reference answer: {_answer_text(record.original_answer)}", "", "Solution outline:", ""]
        body += [f"{i}. {step}" for i, step in enumerate(bp.solution_outline, start=1)]
        if bp.parameters:
            body += ["", "Parameters:", ""]
            body += [
                f"- `{p.slot}` = {p.original}" + (f" — {p.description}" if p.description else "")
                for p in bp.parameters
            ]
        if bp.entities:
            body += ["", "Entities:", ""]
            body += [f"- `{e.slot}` = {e.original} ({e.role})" for e in bp.entities]
        if bp.constraints:
            body += ["", "Constraints:", ""]
            body += [f"- `{c}`" for c in bp.constraints]
        lines += [_details("Answer and blueprint", "\n".join(body)), ""]

    if record.status != "ok":
        lines += [f"**Status: {record.status}**" + (f" — {record.error}" if record.error else ""), ""]
    for note in record.notes:
        lines += [f"> {note}", ""]

    for variant in record.variants:
        lines += [f"### Variant {variant.index}", "", variant.statement_md, ""]
        changes = [
            f"- `{slot}`: {change.original} → {change.variant}"
            for slot, change in {**variant.parameter_map, **variant.entity_map}.items()
        ]
        body = "\n".join([f"**Answer:** {_answer_text(variant.answer)}", "", "Changes:", "", *changes])
        lines += [_details("Answer", body), ""]
    return lines


def to_markdown(results: Results) -> str:
    run = results.run
    models = ", ".join(f"{role} = `{tag}`" for role, tag in run.models.items())
    avg = f"{run.avg_s_per_variant:.1f} s per variant" if run.avg_s_per_variant is not None else "no variants"
    lines = [
        "# MathScrambler results",
        "",
        f"Run started {run.started:%Y-%m-%d %H:%M:%S}, {run.wall_s:.0f} s wall, {avg}.  ",
        f"Seed `{run.seed}`, profile `{run.profile}`, server {run.server}.  ",
        f"Models: {models}.  ",
        f"{run.problems_ok}/{run.problems_total} problems scrambled, "
        f"{run.variants_total} variants ({run.variants_requested} requested per problem).  ",
        f"Verification: {run.verification}.",
        "",
    ]
    if results.skipped:
        lines += ["Skipped inputs:", ""]
        lines += [f"- `{s.path}`: {s.reason}" for s in results.skipped]
        lines.append("")
    for record in results.problems:
        lines += _problem_md(record)
    return "\n".join(lines).rstrip() + "\n"


def write_results(results: Results, out_dir: Path) -> tuple[Path, Path]:
    """Write results.json and results.md into `out_dir`; returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    md_path = out_dir / "results.md"
    json_path.write_text(
        json.dumps(results.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(to_markdown(results), encoding="utf-8")
    return json_path, md_path
