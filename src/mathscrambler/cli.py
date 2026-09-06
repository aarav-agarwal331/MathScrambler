"""The `mathscramble` / `ms` command-line interface (typer).

Commands land phase by phase; only built, working commands are registered —
no stubs (SPEC Section 8).
"""

from __future__ import annotations

import typer
from rich.console import Console

from mathscrambler import __version__, ollama_server, setup_flow
from mathscrambler import doctor as doctor_mod
from mathscrambler.config import ConfigError, load_config

app = typer.Typer(
    name="mathscramble",
    help="Fully local isomorphic math-problem scrambler (Ollama-backed).",
    no_args_is_help=True,
    add_completion=False,
)
server_app = typer.Typer(help="Control the private Ollama server (never touches the global one).")
app.add_typer(server_app, name="server")

console = Console()


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


@server_app.command("start")
def server_start() -> None:
    """Start the private Ollama server (idempotent)."""
    try:
        cfg = load_config()
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    try:
        info = ollama_server.ensure_started(cfg)
    except ollama_server.ServerError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    verb = "started" if info.started_by_us else "already running"
    console.print(f"private server {verb}: {info.base_url} (pid {info.pid}, mode {info.mode})")


@server_app.command("stop")
def server_stop() -> None:
    """Stop the private server — refuses to signal any process it didn't start."""
    result = ollama_server.stop()
    console.print(result.reason)
    raise typer.Exit(0 if result.stopped or "nothing to stop" in result.reason else 1)


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
    app()


if __name__ == "__main__":
    main()
