"""The canonical `Problem` record and the vision extraction schema (Step A).

Everything downstream — blueprint, sampler, verifier, dashboard — sees problems
only through `Problem`, whether they arrived as Markdown, LaTeX, JSON, or a
photograph of a textbook page.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Kind = Literal["computational", "proof", "mixed"]

# Section 3 Step A: extractions below this are flagged for review before scrambling.
LOW_CONFIDENCE = 0.7


class Problem(BaseModel):
    """One problem, from any source. Immutable: ingest is a pure read of the inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    source: str  # the input file it came from, as given on the command line
    source_index: int = 1  # 1-based position within that file (a file may hold several)
    statement_md: str
    given_answer: str | None = None
    tags: list[str] = Field(default_factory=list)
    kind_hint: Kind | None = None  # an author's declaration; the blueprint decides for real
    diagram_description: str | None = None  # images only
    confidence: float | None = None  # images only; None means "not model-extracted"

    @property
    def needs_review(self) -> bool:
        """Vision extractions the user should eyeball before scrambling."""
        return self.confidence is not None and self.confidence < LOW_CONFIDENCE


class ExtractedProblem(BaseModel):
    """One problem as the `vision` role returns it. Schema-constrained decoding."""

    model_config = ConfigDict(extra="forbid")

    statement_md: str
    diagram_description: str = ""
    answer_if_shown: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class VisionExtraction(BaseModel):
    """The whole page. One image may hold several problems — or none."""

    model_config = ConfigDict(extra="forbid")

    problems: list[ExtractedProblem] = Field(default_factory=list)
