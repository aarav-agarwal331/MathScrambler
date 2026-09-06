"""`mathscramble setup` — run-once, idempotent, no sudo (SPEC Section 1.4).

Never edits shell profiles; the only writes are inside the repo, the app-support
directory, and the uv tool venv/shim.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import httpx
import typer
from rich.console import Console

from mathscrambler import doctor, ollama_server, paths
from mathscrambler.config import Config, ConfigError, load_config

KATEX_VERSION = "0.16.11"
KATEX_TARBALL_URL = f"https://registry.npmjs.org/katex/-/katex-{KATEX_VERSION}.tgz"
KATEX_ARTIFACTS = ("katex.min.css", "katex.min.js", "auto-render.min.js", "fonts")
PULL_APPROVAL_BYTES = 30 * 1024**3  # anything larger — or of UNKNOWN size — needs an explicit OK


def katex_dir() -> Path:
    return paths.repo_root() / "src" / "mathscrambler" / "web" / "static" / "katex"


def registry_size(tag_spec: str, timeout: float = 15.0) -> int | None:
    """Download size of a registry tag in bytes; None if it doesn't exist / offline."""
    name, _, tag = tag_spec.partition(":")
    repo = name if "/" in name else f"library/{name}"
    try:
        resp = httpx.get(
            f"https://registry.ollama.ai/v2/{repo}/manifests/{tag or 'latest'}",
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        layers = resp.json().get("layers", [])
        return sum(int(layer.get("size", 0)) for layer in layers) or None
    except (httpx.HTTPError, ValueError):
        return None


def needs_pull_approval(size: int | None, assume_yes: bool) -> bool:
    """SPEC 1.4: ask before pulling anything over 30 GB — and anything whose size
    is unknown, which could be over 30 GB. Unknown size must never disarm the gate."""
    if assume_yes:
        return False
    return size is None or size > PULL_APPROVAL_BYTES


def pull_model(base_url: str, tag: str, console: Console) -> bool:
    """Stream a pull through OUR server; caller must have checked pull_in_progress()."""
    console.print(f"   pulling [bold]{tag}[/bold] ...")
    try:
        with httpx.stream(
            "POST",
            f"{base_url}/api/pull",
            json={"model": tag},
            timeout=httpx.Timeout(30, read=None),
            trust_env=False,
        ) as resp:
            resp.raise_for_status()
            last_pct = -1
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    console.print(f"   [red]pull failed:[/red] unparseable server response: {line[:120]!r}")
                    return False
                if "error" in event:
                    console.print(f"   [red]pull failed:[/red] {event['error']}")
                    return False
                total, done = event.get("total"), event.get("completed")
                if total and done:
                    pct = int(done * 100 / total)
                    if pct // 10 > last_pct // 10:
                        console.print(f"   {tag}: {pct}% of {total / 1e9:.1f} GB")
                        last_pct = pct
        console.print(f"   [green]pulled {tag}[/green]")
        return True
    except (httpx.HTTPError, OSError) as e:
        console.print(f"   [red]pull failed:[/red] {e}")
        return False


def _katex_complete(target: Path) -> bool:
    return all((target / name).exists() for name in KATEX_ARTIFACTS)


def vendor_katex(console: Console) -> None:
    """Download the KaTeX dist into web/static/katex.

    Idempotent and atomic: the check requires EVERY artifact (a partial previous
    attempt is re-done, not skipped), and the files are staged completely before
    replacing the target, so a mid-copy failure can't leave a half-install that
    later runs mistake for done. Failure is a yellow note, never a setup abort.
    """
    target = katex_dir()
    if _katex_complete(target):
        console.print(f" [green]✓[/green] KaTeX {KATEX_VERSION} already vendored at {target}")
        return
    console.print(f"   vendoring KaTeX {KATEX_VERSION} ...")
    try:
        resp = httpx.get(KATEX_TARBALL_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        with tempfile.TemporaryDirectory(dir=paths.scratch_dir()) as tmp:
            tarball = Path(tmp) / "katex.tgz"
            tarball.write_bytes(resp.content)
            with tarfile.open(tarball) as tf:
                tf.extractall(tmp, filter="data")
            dist = Path(tmp) / "package" / "dist"
            staging = Path(tmp) / "staging"
            staging.mkdir()
            shutil.copy2(dist / "katex.min.css", staging / "katex.min.css")
            shutil.copy2(dist / "katex.min.js", staging / "katex.min.js")
            shutil.copy2(dist / "contrib" / "auto-render.min.js", staging / "auto-render.min.js")
            shutil.copytree(dist / "fonts", staging / "fonts")
            if not _katex_complete(staging):
                raise OSError("KaTeX tarball layout unexpected — artifacts missing after extract")
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(target))
    except (httpx.HTTPError, OSError, tarfile.TarError, shutil.Error) as e:
        console.print(f" [yellow]![/yellow] KaTeX vendoring failed ({e}); re-run setup online later")
        return
    console.print(f" [green]✓[/green] KaTeX vendored at {target}")


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def run_setup(console: Console, assume_yes: bool = False) -> int:
    root = paths.repo_root()

    # 1. app-support tree
    paths.ensure_app_dirs()
    console.print(f" [green]✓[/green] app state dir: {paths.app_support_dir()}")

    # 2. project venv + uv tool shim (reinstall keeps the tool venv's deps in sync)
    code, out = _run(["uv", "sync"], cwd=root)
    if code != 0:
        console.print(f" [red]✗[/red] `uv sync` failed:\n{out}")
        return 1
    console.print(" [green]✓[/green] project venv synced (uv)")
    code, out = _run(["uv", "tool", "install", "-e", ".", "--reinstall"], cwd=root)
    if code != 0:
        console.print(f" [red]✗[/red] `uv tool install -e . --reinstall` failed:\n{out}")
        return 1
    console.print(" [green]✓[/green] `mathscramble` shim installed (uv tool, editable)")
    if not shutil.which("mathscramble"):
        bin_dir = doctor._uv_tool_bin_dir()
        console.print(
            f" [yellow]![/yellow] the shim dir is not on your PATH — add this line to your shell profile "
            f'yourself (setup never edits it):\n     export PATH="{bin_dir}:$PATH"'
        )

    # 3. config.toml from the example, only if missing
    if paths.config_path().is_file():
        console.print(" [green]✓[/green] config.toml exists (left untouched)")
    else:
        shutil.copy2(paths.config_example_path(), paths.config_path())
        console.print(" [green]✓[/green] config.toml created from config.example.toml")

    try:
        cfg: Config = load_config()
    except ConfigError as e:
        console.print(f" [red]✗[/red] {e}")
        return 1

    # 4. validate the private server once, then report/pull missing models
    try:
        info = ollama_server.ensure_started(cfg)
        console.print(f" [green]✓[/green] private server validated: {info.base_url} (mode {info.mode})")
    except RuntimeError as e:
        console.print(f" [red]✗[/red] private server failed to start: {e}")
        return 1

    installed = {m.get("name", "") for m in (ollama_server.api_tags(info.port) or [])}
    wanted = sorted(set(cfg.active_roles().resolved_tags().values()))
    missing = [tag for tag in wanted if tag not in installed]
    if not missing:
        console.print(" [green]✓[/green] all configured role models are pulled")
    else:
        for tag in missing:
            size = registry_size(tag)
            size_text = (
                f"{size / 1e9:.1f} GB" if size else "size unknown (registry unreachable or tag missing)"
            )
            console.print(f" [yellow]![/yellow] missing model: {tag} ({size_text})")
            # Re-checked per tag: a pull elsewhere can start while an earlier tag downloads.
            busy = ollama_server.pull_in_progress()
            if busy:
                console.print(f"     skipping pull: {busy}")
                continue
            if needs_pull_approval(size, assume_yes):
                prompt = (
                    f"     {tag} is over 30 GB — pull it now?"
                    if size
                    else f"     {tag} has an unknown size (could exceed 30 GB) — pull it now?"
                )
                if not typer.confirm(prompt, default=False):
                    console.print(f"     skipped {tag} (pull later with `ollama pull {tag}`)")
                    continue
            pull_model(info.base_url, tag, console)

    # 5. KaTeX
    vendor_katex(console)

    console.print("\n running doctor:\n")
    return doctor.render(doctor.run_checks(), console)
