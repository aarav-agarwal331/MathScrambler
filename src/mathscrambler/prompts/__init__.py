"""Prompt files, and the two helpers that read them.

Every prompt lives in this package as a ``.md`` (literal) or ``.jinja``
(templated) file — never as an inline string in Python (Section 8). When a
model misbehaves, the fix belongs in the prompt file or the schema, so the diff
that changed the model's behaviour is legible on its own.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

from jinja2 import StrictUndefined, Template

PROMPT_PACKAGE = "mathscrambler.prompts"


class PromptError(RuntimeError):
    pass


@cache
def _read(filename: str) -> str:
    path = resources.files(PROMPT_PACKAGE).joinpath(filename)
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as e:
        raise PromptError(f"no prompt file {filename!r} in {PROMPT_PACKAGE}") from e


def load(name: str) -> str:
    """The literal text of ``<name>.md``."""
    return _read(f"{name}.md").strip()


def render(name: str, **variables: object) -> str:
    """``<name>.jinja`` rendered with `variables`.

    StrictUndefined: a typo'd variable must fail loudly here, not silently ship
    a prompt with a hole in it.
    """
    template = Template(_read(f"{name}.jinja"), undefined=StrictUndefined)
    return template.render(**variables).strip()
