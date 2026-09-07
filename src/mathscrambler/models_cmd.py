"""`mathscramble models [--pull]` — role/model inventory and explicit pulls.

Listing is read-only (never spawns a server; GETs against whichever of our
usable servers already answers). Pulling goes through the private server with
the same collision guard and >30 GB approval gate as setup.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from mathscrambler import ollama_server, setup_flow
from mathscrambler.config import ROLES, Config


def _installed_tags(cfg: Config) -> tuple[dict[str, int], str] | None:
    """{tag: size_bytes} from the private server if running, else the global one."""
    st = ollama_server.status()
    if st.running and st.record is not None:
        tags = ollama_server.api_tags(st.record.port)
        if tags is not None:
            return {m.get("name", ""): int(m.get("size", 0)) for m in tags}, f"private:{st.record.port}"
    tags = ollama_server.api_tags(cfg.ollama.global_port)
    if tags is not None:
        return {m.get("name", ""): int(m.get("size", 0)) for m in tags}, f"global:{cfg.ollama.global_port}"
    return None


def _residents(cfg: Config) -> dict[str, str]:
    """{tag: "private"|"global"} for currently loaded models (read-only probes)."""
    out: dict[str, str] = {}
    for name in (m.get("name", "") for m in ollama_server.api_ps(cfg.ollama.global_port) or []):
        out[name] = "global"
    st = ollama_server.status()
    if st.running and st.residents:
        for name in (m.get("name", "") for m in st.residents):
            out[name] = "private"  # private wins the label; it's the server we use
    return out


def list_models(cfg: Config, console: Console) -> None:
    queried = _installed_tags(cfg)
    installed = queried[0] if queried else {}
    source = queried[1] if queried else None
    residents = _residents(cfg)

    table = Table(title="configured role models", title_justify="left")
    for col in ("profile", "role", "tag", "status", "resident"):
        table.add_column(col)
    for profile_name in sorted(cfg.roles, key=lambda n: (n != cfg.profile, n)):
        roles_table = cfg.roles[profile_name]
        marker = " (active)" if profile_name == cfg.profile else ""
        for role in ROLES:
            alias = getattr(roles_table, role)
            rc = roles_table.resolve(role)
            if isinstance(alias, str):
                table.add_row(profile_name + marker, role, rc.tag, f"= {alias}", "")
                continue
            if queried is None:
                status = "[dim]unknown (no server)[/dim]"
            elif rc.tag in installed:
                status = f"pulled ({installed[rc.tag] / 1e9:.1f} GB)"
            else:
                status = "[yellow]not pulled[/yellow]"
            table.add_row(profile_name + marker, role, rc.tag, status, residents.get(rc.tag, ""))
    console.print(table)
    if queried is None:
        console.print(
            "[dim]no reachable server to list pulled models — "
            "`mathscramble server start` for live info[/dim]"
        )
    else:
        console.print(f"[dim]pulled/resident info via {source} server (read-only)[/dim]")


def _missing_tags(cfg: Config, installed: set[str]) -> list[str]:
    wanted: set[str] = set()
    for roles_table in cfg.roles.values():
        wanted.update(roles_table.resolved_tags().values())
    return sorted(tag for tag in wanted if tag not in installed)


def pull_missing(cfg: Config, console: Console, assume_yes: bool) -> int:
    """Pull every configured-but-missing tag (all profiles) through the private server."""
    try:
        info = ollama_server.ensure_started(cfg)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    installed = {m.get("name", "") for m in (ollama_server.api_tags(info.port) or [])}
    missing = _missing_tags(cfg, installed)
    if not missing:
        console.print(" [green]✓[/green] all configured role models are already pulled")
        return 0
    failures = 0
    for tag in missing:
        size = setup_flow.registry_size(tag)
        size_text = f"{size / 1e9:.1f} GB" if size else "size unknown (registry unreachable or tag missing)"
        console.print(f" missing: [bold]{tag}[/bold] ({size_text})")
        if setup_flow.needs_pull_approval(size, assume_yes):
            prompt = (
                f"   {tag} is over 30 GB — pull it now?"
                if size
                else f"   {tag} has an unknown size (could exceed 30 GB) — pull it now?"
            )
            if not typer.confirm(prompt, default=False):
                console.print(f"   skipped {tag}")
                continue
        # Collision check runs per tag, AFTER any confirm prompt: a pull elsewhere
        # can start while an earlier tag downloads or while the user deliberates.
        busy = ollama_server.pull_in_progress()
        if busy:
            console.print(f"   skipping: {busy}")
            failures += 1
            continue
        if not setup_flow.pull_model(info.base_url, tag, console):
            failures += 1
    return 1 if failures else 0
