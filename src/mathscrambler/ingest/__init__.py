"""Step A: inputs → canonical `Problem` records.

One entry point for both interfaces: `ingest()` takes whatever the CLI or the
dashboard was handed — files, directories, a mix of text and images — and
returns problems in a stable order, plus an explicit list of what was skipped
and why. Nothing is ever dropped silently.

Inputs are read, never written: no file is opened for writing, renamed, or
given a sibling anywhere under an input path (Section 1.2).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mathscrambler.ingest.images import IMAGE_SUFFIXES, extract_from_image
from mathscrambler.ingest.models import LOW_CONFIDENCE, Problem
from mathscrambler.ingest.text import TEXT_SUFFIXES, IngestError, load_text_file
from mathscrambler.ollama_client import OllamaClient, OllamaClientError

log = logging.getLogger("mathscrambler.ingest")

__all__ = [
    "IMAGE_SUFFIXES",
    "LOW_CONFIDENCE",
    "TEXT_SUFFIXES",
    "Discovered",
    "IdFactory",
    "IngestError",
    "IngestResult",
    "Problem",
    "Skipped",
    "discover",
    "ingest",
    "load_text_inputs",
]

# Directory walks skip repository furniture and prose that is not a problem.
# An explicitly named file is always loaded — the user pointed at that file.
_SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", "outputs", ".venv"})
_SKIP_STEMS = frozenset({"readme", "index", "license", "plan", "spec"})

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Skipped:
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True)
class Discovered:
    files: list[tuple[Path, str]] = field(default_factory=list)  # (path, "text" | "image")
    skipped: list[Skipped] = field(default_factory=list)

    @property
    def text_files(self) -> list[Path]:
        return [p for p, kind in self.files if kind == "text"]

    @property
    def image_files(self) -> list[Path]:
        return [p for p, kind in self.files if kind == "image"]


@dataclass(frozen=True)
class IngestResult:
    problems: list[Problem] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)

    @property
    def needs_review(self) -> list[Problem]:
        """Vision extractions under the confidence bar — flagged, never discarded."""
        return [p for p in self.problems if p.needs_review]


def _classify(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return None


def discover(inputs: Sequence[Path]) -> Discovered:
    """Expand directories and classify inputs, in a stable, sorted order.

    A directory contributes only problem-shaped files; an explicitly named file
    that is not problem-shaped is reported as skipped rather than ignored, so a
    typo'd path never looks like an empty result.
    """
    files: list[tuple[Path, str]] = []
    skipped: list[Skipped] = []
    seen: set[Path] = set()

    def add(path: Path, explicit: bool) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        kind = _classify(path)
        if kind is None:
            if explicit:
                skipped.append(Skipped(path, f"unsupported file type {path.suffix or '(none)'}"))
            return
        seen.add(resolved)
        files.append((path, kind))

    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            skipped.append(Skipped(path, "does not exist"))
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if not child.is_file():
                    continue
                if any(part.startswith(".") or part in _SKIP_DIRS for part in child.parts):
                    continue
                if child.stem.lower() in _SKIP_STEMS:
                    continue
                add(child, explicit=False)
        else:
            add(path, explicit=True)
    return Discovered(files=files, skipped=skipped)


class IdFactory:
    """Stable, collision-free problem ids for one ingest run.

    ``examples/algebra.md`` problem 2 becomes ``algebra-02``; an author's own
    ``<!-- id: ... -->`` wins. A repeated id gets a numeric suffix rather than
    overwriting, because two problems sharing an id would silently collapse in
    the results file.
    """

    def __init__(self) -> None:
        self._used: set[str] = set()

    def __call__(self, path: Path, index: int, declared_id: str | None = None) -> str:
        base = self.slugify(declared_id) if declared_id else f"{self.slugify(path.stem)}-{index:02d}"
        candidate, n = base, 1
        while candidate in self._used:
            n += 1
            candidate = f"{base}-{n}"
        self._used.add(candidate)
        return candidate

    @staticmethod
    def slugify(text: str) -> str:
        slug = _SLUG_RE.sub("-", text.lower()).strip("-")
        return slug or "problem"


def load_text_inputs(inputs: Sequence[Path]) -> IngestResult:
    """Text-shaped inputs only — no model, no server, no network."""
    found = discover(inputs)
    ids = IdFactory()
    problems: list[Problem] = []
    skipped = list(found.skipped)
    for path, kind in found.files:
        if kind != "text":
            skipped.append(Skipped(path, "image input needs the vision model (use ingest())"))
            continue
        try:
            loaded = load_text_file(path, ids)
        except IngestError as e:
            skipped.append(Skipped(path, str(e).removeprefix(f"{path}: ")))
            continue
        if not loaded:
            skipped.append(Skipped(path, "no problems found in file"))
        problems.extend(loaded)
    return IngestResult(problems=problems, skipped=skipped)


async def ingest(
    inputs: Sequence[Path],
    client: OllamaClient | None = None,
    *,
    seed: int | None = None,
) -> IngestResult:
    """Everything in `inputs`, text and images, in discovery order.

    Without a `client`, images are reported as skipped rather than failing the
    run — text problems from the same directory still come through.
    """
    found = discover(inputs)
    ids = IdFactory()
    problems: list[Problem] = []
    skipped = list(found.skipped)
    for path, kind in found.files:
        try:
            if kind == "text":
                loaded = load_text_file(path, ids)
            elif client is None:
                skipped.append(Skipped(path, "no vision client: start the server to read images"))
                continue
            else:
                loaded = await extract_from_image(client, path, ids, seed=seed)
        except (IngestError, OllamaClientError) as e:
            skipped.append(Skipped(path, str(e).removeprefix(f"{path}: ")))
            continue
        if not loaded:
            skipped.append(Skipped(path, "no problems found in file"))
        problems.extend(loaded)
    return IngestResult(problems=problems, skipped=skipped)
