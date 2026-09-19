"""The pipeline, end to end: inputs → problems → blueprint → variants → files.

One code path (Section 1: the CLI today, a dashboard later would call the same
function). Per problem the stages are:

    ingest (text loaders, or the vision model for images)
    → extract_blueprint   reasoner; validated + solver-checked, ≤ N attempts
    → propose_entities    fast; guarded by check_entity_swap
    → sample_variants     deterministic rejection sampling, seeded
    → render              results.json + results.md

What is deliberately *not* here: Step D (an independent re-solve of every
variant). Answers in the output are the blueprint solver's, and the results
say so in as many words.

Everything model-related goes through the private server: `ensure_started`
spawns it if needed, the memory gate refuses before a single weight is loaded,
and the server is left running afterwards so a second run does not pay the
load again (`mathscramble server stop` ends it).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from mathscrambler import ingest, ollama_server, sysinfo
from mathscrambler.config import Config, ProfileRoles, Role
from mathscrambler.entities import propose_entities
from mathscrambler.extract import extract_blueprint
from mathscrambler.ingest.models import Problem
from mathscrambler.ollama_client import OllamaClient, OllamaClientError, TimingRecord
from mathscrambler.render import (
    ProblemRecord,
    ProblemStatus,
    Results,
    RunRecord,
    SamplingRecord,
    SkippedRecord,
    VariantRecord,
    write_results,
)
from mathscrambler.sampler import sample_variants

LOG_TRUNCATE = 4000  # Section 5: prompts and raw responses in run.log, truncated to 4K


class EngineError(RuntimeError):
    """The run could not start or finish; the message says what to do."""


@dataclass(frozen=True)
class RunOptions:
    inputs: list[Path]
    n: int = 3
    seed: int | None = None  # None: drawn at random and recorded, so any run can be replayed
    out_dir: Path | None = None
    profile: str | None = None
    model_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOutcome:
    out_dir: Path
    json_path: Path
    md_path: Path
    results: Results


# --------------------------------------------------------------------------- run.log


class RunLog:
    """Append-only text log of every model exchange and timing in one run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def line(self, text: str) -> None:
        self._fh.write(f"[{datetime.now():%H:%M:%S}] {text}\n")
        self._fh.flush()

    def transcript(self, heading: str, messages: list[dict[str, Any]]) -> None:
        self.line(f"--- {heading} ---")
        for message in messages:
            content = str(message.get("content", ""))
            if len(content) > LOG_TRUNCATE:
                content = content[:LOG_TRUNCATE] + f"… [truncated, {len(content)} chars]"
            self._fh.write(f">>> {message.get('role', '?')}:\n{content}\n\n")
        self._fh.flush()

    def timings(self, label: str, timings: list[TimingRecord]) -> None:
        for t in timings:
            self.line(
                f"{label}: {t.model} {t.total_s:.1f}s total, load {t.load_s:.1f}s, "
                f"{t.prompt_tokens} prompt tok, {t.eval_tokens} eval tok @ {t.tokens_per_s:.1f} tok/s"
            )

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- setup


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def default_out_dir(inputs: list[Path], now: datetime) -> Path:
    """`./outputs/<timestamp>-<slug>/`, the slug from the first input's name (Section 0)."""
    first = inputs[0].name if inputs else "run"
    slug = _SLUG_RE.sub("-", Path(first).stem.lower()).strip("-") or "run"
    return Path.cwd() / "outputs" / f"{now:%Y%m%d-%H%M%S}-{slug}"


def problem_seed(seed: int, problem_id: str) -> int:
    """A per-problem child seed, so retries on one problem never shift another's draws."""
    digest = hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _gate_memory(
    cfg: Config, info: ollama_server.ServerInfo, roles: ProfileRoles, needed: set[Role]
) -> tuple[sysinfo.MemoryGate, list[tuple[str, int]]]:
    """Section 1.2 memory etiquette, for the roles this run will actually load.

    Returns the gate and the load order (largest first, deduplicated by tag —
    the scheduler admits against free memory, so the big model claims its
    slice before the small ones fill the gap).
    """
    installed = {m.get("name", ""): int(m.get("size", 0)) for m in (ollama_server.api_tags(info.port) or [])}
    resident = {m.get("name", "") for m in (ollama_server.api_ps(info.port) or [])}
    tags: dict[str, int] = {}
    for role in needed:
        tag = roles.resolve(role).tag
        if tag not in installed:
            raise EngineError(f"{role} model {tag!r} is not pulled — run `mathscramble models --pull`")
        tags[tag] = installed[tag]
    weight = sum(size for tag, size in tags.items() if tag not in resident)
    global_ps = ollama_server.api_ps(cfg.ollama.global_port) or []
    global_residents = [
        (f"global ollama: {m.get('name', '?')}", int(m.get("size_vram", m.get("size", 0)))) for m in global_ps
    ]
    gate = sysinfo.memory_gate(
        sysinfo.memory_info(),
        sysinfo.projected_footprint(weight),
        cfg.memory.max_fraction,
        global_residents,
        sysinfo.detect_comfyui(),
    )
    order = sorted(tags.items(), key=lambda item: -item[1])
    return gate, order


# --------------------------------------------------------------------------- per problem


async def scramble_problem(
    client: OllamaClient,
    cfg: Config,
    problem: Problem,
    *,
    n: int,
    seed: int,
    log: RunLog,
    console: Console,
) -> ProblemRecord:
    """Blueprint, scenery, variants for one problem. Never raises for model output."""
    timings: dict[str, float] = {}
    label = f"[bold]{problem.id}[/bold]"

    started = time.monotonic()
    extraction = await extract_blueprint(
        client,
        problem,
        max_retries=cfg.sampling.blueprint_max_retries,
        timeout_s=cfg.sandbox.timeout_s,
        memory_mb=cfg.sandbox.memory_mb,
    )
    timings["blueprint"] = time.monotonic() - started
    log.transcript(f"{problem.id}: blueprint ({extraction.attempts} attempt(s))", extraction.transcript)
    log.timings(f"{problem.id}: blueprint", extraction.timings)
    if not extraction.ok or extraction.blueprint is None:
        console.print(f"  {label} blueprint [red]failed[/red] after {extraction.attempts} attempt(s)")
        for why in extraction.problems:
            console.print(f"      [dim]{why}[/dim]")
        return ProblemRecord(
            original=problem,
            status="blueprint_failed",
            error="; ".join(extraction.problems)[:1000],
            blueprint_attempts=extraction.attempts,
            timings_s=timings,
        )
    bp = extraction.blueprint
    tok_s = extraction.timings[-1].tokens_per_s if extraction.timings else 0.0
    answer_text = extraction.answer.text if extraction.answer else "-"
    console.print(
        f"  {label} blueprint [green]ok[/green] attempt {extraction.attempts} "
        f"({timings['blueprint']:.0f} s, {tok_s:.0f} tok/s, answer {answer_text})"
    )

    if bp.solver_code is None:
        note = (
            "proof-kind problem: the blueprint was extracted, but generating and checking proof "
            "variants needs the verifier (Step D), which is not implemented yet"
        )
        console.print(f"  {label} [yellow]skipped[/yellow]: {note}")
        return ProblemRecord(
            original=problem,
            status="proof_unsupported",
            error=note,
            blueprint=bp,
            blueprint_attempts=extraction.attempts,
            timings_s=timings,
        )

    started = time.monotonic()
    scenery = await propose_entities(client, bp, n=n)
    timings["entities"] = time.monotonic() - started
    log.transcript(f"{problem.id}: entities", scenery.transcript)
    log.timings(f"{problem.id}: entities", scenery.timings)
    for why in scenery.rejected:
        log.line(f"{problem.id}: entity proposal rejected: {why}")
    notes: list[str] = []
    if bp.entities and not scenery.complete:
        notes.append(
            f"only {scenery.accepted} distinct scenery proposals passed the guard for {n} variants; "
            + ("some variants share scenery" if scenery.accepted else "variants keep the original scenery")
        )
    console.print(
        f"  {label} scenery {scenery.accepted}/{n} proposals accepted "
        f"({timings['entities']:.0f} s, {len(scenery.rejected)} rejected)"
    )

    started = time.monotonic()
    assert extraction.answer is not None  # check_solver ran the solver: a solver blueprint has an answer
    report = await asyncio.to_thread(
        sample_variants,
        bp,
        extraction.answer,
        n=n,
        seed=seed,
        max_samples=cfg.sampling.max_samples,
        relaxed_extra_samples=cfg.sampling.relaxed_extra_samples,
        keep_nice_answers=cfg.sampling.keep_nice_answers,
        reject_degenerate=cfg.sampling.reject_degenerate,
        entity_maps=scenery.maps,
        timeout_s=cfg.sandbox.timeout_s,
        memory_mb=cfg.sandbox.memory_mb,
    )
    timings["sampling"] = time.monotonic() - started
    failed = f"; FAILED: {report.failure}" if report.failure else ""
    log.line(f"{problem.id}: sampling {report.summary()}{failed}")
    variants = [VariantRecord.from_variant(bp, v) for v in report.variants]
    status: ProblemStatus
    if len(variants) == n:
        status = "ok"
    elif variants:
        status = "partial"
    else:
        status = "sampling_failed"
    colour = {"ok": "green", "partial": "yellow", "sampling_failed": "red"}[status]
    console.print(f"  {label} sampling [{colour}]{report.summary()}[/{colour}] ({timings['sampling']:.1f} s)")
    return ProblemRecord(
        original=problem,
        status=status,
        error=report.failure,
        blueprint=bp,
        blueprint_attempts=extraction.attempts,
        original_answer=extraction.answer,
        entities_accepted=scenery.accepted,
        sampling=SamplingRecord(
            attempts=report.attempts, rejections=dict(report.rejections), failure=report.failure
        ),
        variants=variants,
        timings_s=timings,
        notes=notes,
    )


# --------------------------------------------------------------------------- the run


async def run(cfg: Config, options: RunOptions, console: Console) -> RunOutcome:
    started_at = datetime.now()
    wall_started = time.monotonic()
    seed = options.seed if options.seed is not None else secrets.randbelow(2**31)
    roles = cfg.active_roles(options.profile, options.model_overrides)
    profile = options.profile or cfg.profile

    found = ingest.discover(options.inputs)
    if not found.files:
        reasons = "; ".join(str(s) for s in found.skipped) or "no problem-shaped files under the given paths"
        raise EngineError(f"nothing to scramble: {reasons}")
    needed: set[Role] = {"reasoner", "fast"}
    if found.image_files:
        needed.add("vision")

    out_dir = options.out_dir or default_out_dir(options.inputs, started_at)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    try:
        return await _run(
            cfg, options, console, log, roles, profile, seed, needed, found, out_dir, started_at, wall_started
        )
    finally:
        log.close()


async def _run(
    cfg: Config,
    options: RunOptions,
    console: Console,
    log: RunLog,
    roles: ProfileRoles,
    profile: str,
    seed: int,
    needed: set[Role],
    found: ingest.Discovered,
    out_dir: Path,
    started_at: datetime,
    wall_started: float,
) -> RunOutcome:
    log.line(
        f"run started; inputs={[str(p) for p in options.inputs]} n={options.n} seed={seed} profile={profile}"
    )
    console.print(f"run  [bold]{out_dir}[/bold]  seed {seed}  profile {profile}")

    try:
        info = ollama_server.ensure_started(cfg)
    except RuntimeError as e:  # ServerError, port exhaustion, config problems
        raise EngineError(str(e)) from e
    server_label = f"{info.mode} {info.base_url}"
    log.line(f"server: {server_label} (pid {info.pid}, started by this run: {info.started_by_us})")
    console.print(f"server  {server_label}" + (" (started)" if info.started_by_us else " (already running)"))

    gate, load_order = _gate_memory(cfg, info, roles, needed)
    log.line(gate.message)
    if not gate.ok:
        if info.started_by_us:
            ollama_server.stop()
        raise EngineError(gate.message)

    async with OllamaClient(info.base_url, roles, keep_alive=cfg.ollama.keep_alive) as client:
        by_tag = {roles.resolve(role).tag: role for role in sorted(needed)}
        for tag, size in load_order:
            role = by_tag[tag]
            console.print(f"load  {role} [bold]{tag}[/bold] ({size / 1e9:.1f} GB) …", end=" ")
            # Wall clock, not the reply's counters: a load-only call reports no timings.
            started = time.monotonic()
            try:
                await client.preload(role)
            except OllamaClientError as e:
                raise EngineError(f"could not load {tag}: {e}") from e
            elapsed = time.monotonic() - started
            console.print(f"{elapsed:.1f} s")
            log.line(f"preload {role}={tag}: {elapsed:.1f}s")

        ingested = await ingest.ingest(options.inputs, client)
        for skipped in ingested.skipped:
            log.line(f"skipped input: {skipped}")
            console.print(f"skip  [dim]{skipped}[/dim]")
        if ingested.needs_review:
            ids = ", ".join(p.id for p in ingested.needs_review)
            console.print(f"[yellow]low-confidence vision extraction (check the statements): {ids}[/yellow]")
        console.print(f"ingest  {len(ingested.problems)} problem(s) from {len(found.files)} file(s)")

        records: list[ProblemRecord] = []
        for index, problem in enumerate(ingested.problems, start=1):
            console.print(
                f"[{index}/{len(ingested.problems)}] {problem.id}: {problem.statement_md[:70].strip()}…"
            )
            record = await scramble_problem(
                client,
                cfg,
                problem,
                n=options.n,
                seed=problem_seed(seed, problem.id),
                log=log,
                console=console,
            )
            records.append(record)

    finished_at = datetime.now()
    wall = time.monotonic() - wall_started
    variants_total = sum(len(r.variants) for r in records)
    results = Results(
        run=RunRecord(
            started=started_at,
            finished=finished_at,
            wall_s=wall,
            seed=seed,
            profile=profile,
            models={role: roles.resolve(role).tag for role in ("reasoner", "fast", "vision")},
            server=server_label,
            inputs=[str(p) for p in options.inputs],
            variants_requested=options.n,
            problems_total=len(records),
            problems_ok=sum(r.status == "ok" for r in records),
            variants_total=variants_total,
            avg_s_per_variant=(wall / variants_total) if variants_total else None,
        ),
        problems=records,
        skipped=[SkippedRecord(path=str(s.path), reason=s.reason) for s in ingested.skipped],
    )
    json_path, md_path = write_results(results, out_dir)
    log.line(f"wrote {json_path} and {md_path}; {variants_total} variants in {wall:.0f}s")
    return RunOutcome(out_dir=out_dir, json_path=json_path, md_path=md_path, results=results)
