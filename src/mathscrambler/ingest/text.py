"""Text-shaped loaders: Markdown, plain text, LaTeX, JSON.

All four are pure reads. Nothing in this module opens an input file for
writing, renames it, or writes a sibling file — inputs are never mutated
(Section 1.2).

Splitting rule (shared with the dashboard's paste box): a line of three or more
dashes on its own separates problems. Separators inside a fenced code block are
literal, not separators.

Per-problem metadata rides in comments — HTML in Markdown, `%` in LaTeX — so it
is invisible when the source is rendered and never pollutes the statement the
model sees::

    <!-- answer: 8 -->        % answer: 8
    <!-- tags: geometry -->   % tags: geometry
    <!-- kind: proof -->      % kind: proof
    <!-- id: rect-area -->    % id: rect-area

Only those four keys are consumed; any other comment is left in the statement
untouched, because silently deleting an author's text is worse than ignoring it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mathscrambler.ingest.models import Problem

TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".text", ".tex", ".json"})

_SEPARATOR_RE = re.compile(r"^\s*-{3,}\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
# `<!-- answer: 8 -->` in Markdown, `% answer: 8` in LaTeX — each invisible in
# its own format, so a directive never shows up in the rendered statement.
_DIRECTIVE_RE = re.compile(
    r"^(?:<!--\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*-->|%\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?))\s*$"
)
_DIRECTIVE_KEYS = frozenset({"answer", "tags", "kind", "id"})
# A front-matter line: `key: value` (bare key, no spaces) or a `- ` list item.
_YAML_LINE_RE = re.compile(r"^\s*(?:[\w.-]+\s*:|-\s)")
_KINDS = frozenset({"computational", "proof", "mixed"})

# \begin{problem} ... \end{problem} and friends, non-greedy, across lines.
_TEX_ENV_RE = re.compile(
    r"\\begin\{(problem|exercise|question)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}",
    re.DOTALL,
)
_TEX_DOCUMENT_RE = re.compile(r"\\begin\{document\}(.*?)\\end\{document\}", re.DOTALL)


class IngestError(RuntimeError):
    """An input could not be read as problems. Carries the path in its message."""


# --------------------------------------------------------------------------- splitting


def strip_front_matter(text: str) -> str:
    """Drop a leading YAML front-matter block.

    `---` opens front matter only at the very top of a file; anywhere else it is
    our problem separator. Without this a document's title and author get
    ingested as a maths question.

    The block must also *look* like YAML — every line a `key: value` or a `-`
    list item. A leading `---` followed by prose is an empty first chunk before
    a real problem, and eating that would delete the author's work.
    """
    lines = text.splitlines()
    if not lines or not _SEPARATOR_RE.match(lines[0]):
        return text
    for i, line in enumerate(lines[1:], start=1):
        if _SEPARATOR_RE.match(line):
            block = [ln for ln in lines[1:i] if ln.strip()]
            if block and all(_YAML_LINE_RE.match(ln) for ln in block):
                return "\n".join(lines[i + 1 :])
            return text
    return text  # never closed: not front matter, just a leading separator


def split_chunks(text: str) -> list[str]:
    """Split on `---` lines, ignoring separators inside fenced code blocks."""
    chunks: list[str] = []
    current: list[str] = []
    fence: str | None = None
    for line in strip_front_matter(text).splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            fence = None if fence == marker else (fence or marker)
        elif fence is None and _SEPARATOR_RE.match(line):
            chunks.append("\n".join(current))
            current = []
            continue
        current.append(line)
    chunks.append("\n".join(current))
    return [c.strip() for c in chunks if c.strip()]


def parse_directives(chunk: str) -> tuple[str, dict[str, str]]:
    """Strip recognized ``<!-- key: value -->`` lines; return (statement, directives).

    Fence-aware, for the same reason `split_chunks` is: a directive-shaped line
    inside a code block is content a problem may be *about*, and consuming it
    would both delete it from the statement and set metadata the author never wrote.
    """
    kept: list[str] = []
    found: dict[str, str] = {}
    fence: str | None = None
    for line in chunk.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            fence = None if fence == marker else (fence or marker)
            kept.append(line)
            continue
        if fence is not None:
            kept.append(line)
            continue
        match = _DIRECTIVE_RE.match(line.strip())
        if match:
            html_key, html_value, tex_key, tex_value = match.groups()
            key, value = (html_key, html_value) if html_key else (tex_key, tex_value)
            if key and key.lower() in _DIRECTIVE_KEYS:
                found[key.lower()] = value or ""
                continue
        kept.append(line)
    return "\n".join(kept).strip(), found


def _apply_directives(fields: dict[str, Any], directives: dict[str, str], where: str) -> None:
    if answer := directives.get("answer"):
        fields["given_answer"] = answer
    if tags := directives.get("tags"):
        fields["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if kind := directives.get("kind"):
        if kind not in _KINDS:
            raise IngestError(f"{where}: kind must be one of {sorted(_KINDS)}, got {kind!r}")
        fields["kind_hint"] = kind
    if declared_id := directives.get("id"):
        fields["declared_id"] = declared_id


# --------------------------------------------------------------------------- loaders


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise IngestError(f"{path}: not UTF-8 text ({e.reason})") from e
    except OSError as e:
        raise IngestError(f"{path}: cannot read ({e.strerror})") from e


def _chunks_to_fields(chunks: list[str], path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        statement, directives = parse_directives(chunk)
        if not statement:
            continue  # a chunk of nothing but metadata is not a problem
        fields: dict[str, Any] = {"statement_md": statement, "source_index": index}
        _apply_directives(fields, directives, f"{path} problem {index}")
        out.append(fields)
    return out


def load_markdown(path: Path) -> list[dict[str, Any]]:
    return _chunks_to_fields(split_chunks(_read(path)), path)


def _uncommented(body: str) -> str:
    """Blank out whole-line LaTeX comments, keeping our own directive lines.

    Commenting a problem out is how an author removes it from a `.tex` file; a
    `% \\begin{problem}` that still extracted would resurrect deleted work.
    Line-level only — an inline `x % note` is left alone — and `\\%` is not a
    comment, so an escaped percent is untouched.
    """
    kept: list[str] = []
    for line in body.splitlines():
        stripped = line.lstrip()
        is_comment = stripped.startswith("%")
        match = _DIRECTIVE_RE.match(stripped)
        is_directive = bool(match) and (match.group(3) or "").lower() in _DIRECTIVE_KEYS
        kept.append("" if is_comment and not is_directive else line)
    return "\n".join(kept)


def load_tex(path: Path) -> list[dict[str, Any]]:
    """`\\begin{problem}` environments if the file uses them, else `---` splitting.

    The statement stays LaTeX: downstream renders it with KaTeX, and rewriting
    it into Markdown here would lose notation the blueprint needs.
    """
    text = _read(path)
    body = match.group(1) if (match := _TEX_DOCUMENT_RE.search(text)) else text
    envs = [m.group(2) for m in _TEX_ENV_RE.finditer(_uncommented(body))]
    return _chunks_to_fields([e.strip() for e in envs] if envs else split_chunks(body), path)


def _json_item_fields(item: Any, path: Path, index: int) -> dict[str, Any]:
    where = f"{path} problem {index}"
    if isinstance(item, str):
        if not item.strip():
            raise IngestError(f"{where}: empty statement")
        return {"statement_md": item.strip(), "source_index": index}
    if not isinstance(item, dict):
        raise IngestError(f"{where}: expected an object or a string, got {type(item).__name__}")
    key = next((k for k in ("statement_md", "statement", "problem", "text") if item.get(k)), None)
    if key is None:
        raise IngestError(f"{where}: no statement (expected one of statement_md/statement/problem/text)")
    # str() on a list or dict would silently ship a Python repr as the problem text.
    if not isinstance(item[key], str):
        raise IngestError(f"{where}: {key} must be a string, got {type(item[key]).__name__}")
    statement = item[key]
    if not statement.strip():
        raise IngestError(f"{where}: {key} is empty")
    fields: dict[str, Any] = {"statement_md": statement.strip(), "source_index": index}
    answer = next((item[k] for k in ("given_answer", "answer") if item.get(k) is not None), None)
    if answer is not None:
        fields["given_answer"] = str(answer)
    tags = item.get("tags")
    if isinstance(tags, list):
        fields["tags"] = [str(t) for t in tags]
    elif isinstance(tags, str):
        fields["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    kind = item.get("kind") or item.get("kind_hint")
    if kind:
        if kind not in _KINDS:
            raise IngestError(f"{where}: kind must be one of {sorted(_KINDS)}, got {kind!r}")
        fields["kind_hint"] = str(kind)
    if declared_id := item.get("id"):
        fields["declared_id"] = str(declared_id)
    return fields


def load_json(path: Path) -> list[dict[str, Any]]:
    """A list of problems, or ``{"problems": [...]}``; items may be objects or strings."""
    try:
        data = json.loads(_read(path))
    except json.JSONDecodeError as e:
        raise IngestError(f"{path}: invalid JSON at line {e.lineno}: {e.msg}") from e
    if isinstance(data, dict):
        items = data.get("problems")
        if items is None:
            raise IngestError(f"{path}: object has no 'problems' key")
    else:
        items = data
    if not isinstance(items, list):
        raise IngestError(f"{path}: expected a list of problems, got {type(items).__name__}")
    return [_json_item_fields(item, path, i) for i, item in enumerate(items, start=1)]


_LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_markdown,
    ".text": load_markdown,
    ".tex": load_tex,
    ".json": load_json,
}


def load_text_file(path: Path, id_factory: Any) -> list[Problem]:
    """Every problem in one text-shaped file, as `Problem` records.

    `id_factory(path, index, declared_id)` supplies collision-free ids; ingest
    owns that so ids stay stable and unique across a whole run.
    """
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise IngestError(f"{path}: unsupported text suffix {path.suffix!r}")
    problems: list[Problem] = []
    for fields in loader(path):
        declared_id = fields.pop("declared_id", None)
        problems.append(
            Problem(
                id=id_factory(path, fields["source_index"], declared_id),
                source=str(path),
                **fields,
            )
        )
    return problems
