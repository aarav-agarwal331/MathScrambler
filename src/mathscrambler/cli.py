"""The `mathscramble` / `ms` command-line interface (typer).

Commands land phase by phase; only built, working commands are registered —
no stubs (SPEC Section 8).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from mathscrambler import __version__, engine, models_cmd, ollama_server, setup_flow
from mathscrambler import doctor as doctor_mod
from mathscrambler.config import ConfigError, load_config, parse_model_overrides

app = typer.Typer(
    name="mathscramble",
    help="Fully local isomorphic math-problem scrambler (Ollama-backed).",
    no_args_is_help=True,
    add_completion=False,
)
server_app = typer.Typer(help="Control the private Ollama server (never touches the global one).")
app.add_typer(server_app, name="server")

console = Console(highlight=False)  # prose, not reprs: no auto-colouring of digits and brackets


@app.callback(invoke_without_command=True)
def _main(
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        console.print(f"mathscramble {__version__}")
        raise typer.Exit(0)


@app.command()
def setup(
    yes: bool = typer.Option(False, "--yes", help="Assume yes for >30 GB pull confirmations."),
) -> None:
    """One-time environment setup: dirs, venv + shim, config, server validation, KaTeX."""
    raise typer.Exit(setup_flow.run_setup(console, assume_yes=yes))


@app.command()
def doctor() -> None:
    """Environment health report (green/yellow/red; every red comes with its fix)."""
    raise typer.Exit(doctor_mod.render(doctor_mod.run_checks(), console))


@app.command()
def models(
    pull: bool = typer.Option(False, "--pull", help="Pull missing configured models (private server)."),
    yes: bool = typer.Option(False, "--yes", help="Assume yes for >30 GB pull confirmations."),
) -> None:
    """List configured role models (pulled/size/resident); --pull fetches missing ones."""
    try:
        cfg = load_config()
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    models_cmd.list_models(cfg, console)
    if pull:
        raise typer.Exit(models_cmd.pull_missing(cfg, console, assume_yes=yes))


@app.command()
def run(
    inputs: Annotated[
        list[Path],
        typer.Argument(help="Problem files or folders: .md/.txt/.tex/.json, or images (png/jpg/...)."),
    ],
    n: Annotated[int, typer.Option("-n", "--variants", min=1, max=10, help="Variants per problem.")] = 3,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output folder (default ./outputs/<timestamp>-<slug>/).")
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Sampling seed; omitted = random, recorded in results.")
    ] = None,
    lite: Annotated[
        bool, typer.Option("--lite", help="Use the lite role profile (smaller reasoner).")
    ] = False,
    models: Annotated[
        str | None,
        typer.Option(
            "--models", help="Per-role tag overrides, e.g. reasoner=gpt-oss:120b,vision=qwen2.5vl:7b"
        ),
    ] = None,
) -> None:
    """Scramble every problem under INPUTS into isomorphic variants (results.json + results.md)."""
    try:
        cfg = load_config()
        options = engine.RunOptions(
            inputs=list(inputs),
            n=n,
            seed=seed,
            out_dir=out,
            profile="lite" if lite else None,
            model_overrides=parse_model_overrides(models),
        )
        outcome = asyncio.run(engine.run(cfg, options, console))
    except (ConfigError, engine.EngineError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print(
            "[yellow]interrupted[/yellow] — the private server is still up; `mathscramble server stop`"
        )
        raise typer.Exit(130) from None
    summary = outcome.results.run
    console.print(
        f"done  {summary.problems_ok}/{summary.problems_total} problems scrambled, "
        f"{summary.variants_total} variants in {summary.wall_s:.0f} s"
        + (f" ({summary.avg_s_per_variant:.0f} s per variant)" if summary.avg_s_per_variant else "")
    )
    for record in outcome.results.problems:
        if not record.variants:
            continue
        reference = record.original_answer.text if record.original_answer else "?"
        console.print(f"\n[bold]{record.original.id}[/bold]  [dim]original answer {escape(reference)}[/dim]")
        for variant in record.variants:
            statement = " ".join(variant.statement_md.split())
            console.print(f"  [dim]{variant.index}.[/dim] {escape(statement)}")
            console.print(f"     [dim]answer[/dim] {escape(variant.answer.text)}")
    console.print()
    console.print(f"results  {outcome.md_path}")
    console.print(f"         {outcome.json_path}")
    console.print("[dim]the private server stays up for keep_alive; `mathscramble server stop` ends it[/dim]")
    raise typer.Exit(0 if summary.problems_ok == summary.problems_total else 2)


@server_app.command("start")
def server_start() -> None:
    """Start the private Ollama server (idempotent)."""
    try:
        cfg = load_config()
        info = ollama_server.ensure_started(cfg)
    except RuntimeError as e:  # ConfigError, ServerError, port exhaustion, repo discovery
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    verb = "started" if info.started_by_us else "already running"
    console.print(f"private server {verb}: {info.base_url} (pid {info.pid}, mode {info.mode})")


@server_app.command("stop")
def server_stop() -> None:
    """Stop the private server — refuses to signal any process it didn't start."""
    result = ollama_server.stop()
    console.print(result.reason)
    # ok = the desired end state holds (no private server of ours running),
    # even when there was nothing to signal or the record was stale.
    raise typer.Exit(0 if result.ok else 1)


@server_app.command("status")
def server_status() -> None:
    """Show private-server status (PID, port, resident models)."""
    st = ollama_server.status()
    console.print(st.detail)
    if st.running and st.residents is not None:
        names = ", ".join(m.get("name", "?") for m in st.residents) or "none"
        console.print(f"resident models: {names}")
    raise typer.Exit(0 if st.running else 1)


def main() -> None:
    # Surface module warnings (e.g. shared-mode fallback, owned-but-wedged restarts).
    logging.basicConfig(level=logging.WARNING, format="warning: %(message)s")
    app()


if __name__ == "__main__":
    main()
