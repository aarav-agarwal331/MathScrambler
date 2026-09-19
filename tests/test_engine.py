"""The run path, wired: problem → blueprint → scenery → variants → files.

Model replies are scripted on the HTTP stub; everything else — the client, the
validators, the solver sandbox, the sampler, the renderer — is the real code.
Live behaviour against Ollama is checked by hand (`mathscramble run`), not here.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from conftest import PEN_BLUEPRINT
from mathscrambler import engine
from mathscrambler.cli import app
from mathscrambler.config import Config, ProfileRoles, RoleConfig
from mathscrambler.engine import RunLog, problem_seed, scramble_problem
from mathscrambler.ingest.models import Problem
from mathscrambler.ollama_client import OllamaClient
from mathscrambler.render import Results, RunRecord, to_markdown, write_results

REASONER = RoleConfig(tag="gpt-oss:test", num_ctx=16384, temperature=0.3, reasoning_effort="high")
VISION = RoleConfig(tag="qwen-test:v", num_ctx=8192, temperature=0.0, think=False)
ROLES = ProfileRoles(vision=VISION, reasoner=REASONER, fast="vision")

PEN_PROBLEM = Problem(
    id="worksheet-01",
    source="examples/images/page-worksheet.png",
    statement_md="A shop sells pens at $3$ for $\\$2$. At the same rate, how much do $21$ pens cost?",
    given_answer="$14",
)

PROPOSALS = {
    "proposals": [
        {
            "replacements": [
                {"slot": "E1", "replacement": "market stall"},
                {"slot": "E2", "replacement": "erasers"},
            ]
        },
        {
            "replacements": [
                {"slot": "E1", "replacement": "kiosk"},
                {"slot": "E2", "replacement": "notebooks"},
            ]
        },
        {
            "replacements": [{"slot": "E1", "replacement": "shop"}, {"slot": "E2", "replacement": "pens"}]
        },  # unchanged
        {
            "replacements": [
                {"slot": "E1", "replacement": "bakery"},
                {"slot": "E2", "replacement": "3 rolls"},
            ]
        },  # digit
        {
            "replacements": [
                {"slot": "E1", "replacement": "bookshop"},
                {"slot": "E2", "replacement": "pencils"},
            ]
        },
    ]
}


def _chat_body(content: str) -> dict:
    return {
        "model": "stub",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "total_duration": 2_000_000_000,
        "eval_count": 40,
        "eval_duration": 1_000_000_000,
    }


def _config() -> Config:
    return Config(roles={"default": ROLES})


async def test_scramble_problem_end_to_end(tmp_home: Path, tmp_path: Path, http_stub):
    # Call order on the stub: blueprint (reasoner), then scenery proposals (fast).
    http_stub.post_route(
        "/api/chat",
        _chat_body(json.dumps(PEN_BLUEPRINT)),
        _chat_body(json.dumps(PROPOSALS)),
    )
    log = RunLog(tmp_path / "run.log")
    async with OllamaClient(f"http://127.0.0.1:{http_stub.port}", ROLES) as client:
        record = await scramble_problem(
            client,
            _config(),
            PEN_PROBLEM,
            n=3,
            seed=problem_seed(42, PEN_PROBLEM.id),
            log=log,
            console=Console(file=io.StringIO(), width=120),
        )
    log.close()

    assert record.status == "ok", record.error
    assert record.blueprint_attempts == 1
    assert record.original_answer is not None and record.original_answer.text == "14"
    assert len(record.variants) == 3
    # The two proposals that broke the guard were rejected; the three good ones were used, in order.
    assert record.entities_accepted == 3
    assert [v.entity_map["E2"].variant for v in record.variants] == ["erasers", "notebooks", "pencils"]
    for variant in record.variants:
        # Every number changed, the answer is still a whole number, and the statement was re-rendered.
        for slot, change in variant.parameter_map.items():
            assert change.original != change.variant, slot
        assert variant.answer.kind == "integer"
        assert "{p1}" not in variant.statement_md
        assert variant.entity_map["E2"].variant in variant.statement_md
        assert str(variant.parameter_map["p3"].variant) in variant.statement_md
    # Sandbox check: the solver's arithmetic matches the rendered numbers.
    first = record.variants[0]
    p1, p2, p3 = (int(first.parameter_map[s].variant) for s in ("p1", "p2", "p3"))
    assert int(first.answer.text) == p3 // p1 * p2

    # The transcript reached run.log: both prompts and both replies, truncated-safe.
    text = (tmp_path / "run.log").read_text()
    assert "blueprint (1 attempt(s))" in text
    assert ">>> system:" in text and "template_md" in text
    assert "entity proposal rejected" in text

    # Rendering: JSON round-trips through the schema, Markdown carries the honesty marker.
    results = Results(
        run=RunRecord(
            started=datetime.now(),
            finished=datetime.now(),
            wall_s=1.0,
            seed=42,
            profile="default",
            models={"reasoner": "gpt-oss:test", "fast": "qwen-test:v", "vision": "qwen-test:v"},
            server="private http://127.0.0.1:0",
            inputs=["examples/"],
            variants_requested=3,
            problems_total=1,
            problems_ok=1,
            variants_total=3,
            avg_s_per_variant=0.3,
        ),
        problems=[record],
    )
    json_path, md_path = write_results(results, tmp_path / "out")
    reloaded = Results.model_validate_json(json_path.read_text())
    assert reloaded.problems[0].variants[2].statement_md == record.variants[2].statement_md
    md = md_path.read_text()
    assert md == to_markdown(results)
    assert "### Variant 3" in md
    assert "not implemented (Step D)" in md
    assert PEN_PROBLEM.statement_md in md


async def test_blueprint_rejected_then_fixed(tmp_home: Path, tmp_path: Path, http_stub):
    """A blueprint that fails the mechanical gate is sent back with the faults, once."""
    broken = dict(PEN_BLUEPRINT, solver_code="def solve(p1, p2, p3):\n    return p3 * p2\n")  # 42, not 14
    http_stub.post_route(
        "/api/chat",
        _chat_body(json.dumps(broken)),
        _chat_body(json.dumps(PEN_BLUEPRINT)),
        _chat_body(json.dumps(PROPOSALS)),
    )
    log = RunLog(tmp_path / "run.log")
    async with OllamaClient(f"http://127.0.0.1:{http_stub.port}", ROLES) as client:
        record = await scramble_problem(
            client, _config(), PEN_PROBLEM, n=1, seed=1, log=log, console=Console(file=io.StringIO())
        )
    log.close()
    assert record.status == "ok"
    assert record.blueprint_attempts == 2
    feedback = [b for path, b in http_stub.posts if path == "/api/chat"][1]["messages"][-1]["content"]
    assert "rejected" in feedback and "does not reproduce" in feedback


async def test_proof_blueprint_is_reported_not_scrambled(tmp_home: Path, tmp_path: Path, http_stub):
    proof = {
        "kind": "proof",
        "entities": [],
        "parameters": [
            {"slot": "p1", "original": 7, "type": "int", "description": "how many integers"},
            {"slot": "p2", "original": 6, "type": "int", "description": "the modulus"},
        ],
        "constraints": ["p1 > p2", "p2 > 1"],
        "template_md": "Prove that among any ${p1}$ integers, two differ by a multiple of ${p2}$.",
        "solution_outline": ["pigeonhole on residues"],
        "solver_code": None,
        "answer_type": "proof",
    }
    http_stub.post_route("/api/chat", _chat_body(json.dumps(proof)))
    log = RunLog(tmp_path / "run.log")
    problem = Problem(id="pigeon", source="proofs.md", statement_md="Prove ...", kind_hint="proof")
    async with OllamaClient(f"http://127.0.0.1:{http_stub.port}", ROLES) as client:
        record = await scramble_problem(
            client, _config(), problem, n=2, seed=1, log=log, console=Console(file=io.StringIO())
        )
    log.close()
    assert record.status == "proof_unsupported"
    assert record.blueprint is not None and record.blueprint.kind == "proof"
    assert record.variants == []
    assert "Step D" in (record.error or "")


def test_run_command_is_registered_and_refuses_empty_inputs(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    assert runner.invoke(app, ["run", "--help"]).exit_code == 0
    monkeypatch.setattr(engine, "run", None)  # must never be reached without inputs
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
