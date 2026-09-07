"""Image → problems via the `vision` role (Section 3, Step A).

The image is normalized in memory only — EXIF rotation applied, RGB, long side
capped — and the original file is never rewritten or copied next to itself.
A phone photo of a textbook page is the motivating input: unrotated EXIF makes
a legible page unreadable to the model, and a 12 MP original wastes minutes of
prompt-eval for detail the model cannot use.
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageOps

from mathscrambler import prompts
from mathscrambler.ingest.models import Problem, VisionExtraction
from mathscrambler.ingest.text import IngestError
from mathscrambler.ollama_client import OllamaClient

log = logging.getLogger("mathscrambler.ingest.images")

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})

# Long-side cap. Textbook text stays legible well below phone-camera resolution,
# and every extra pixel is prompt-eval time on a model that cannot use it.
MAX_IMAGE_PX = 1600


def prepare_image_bytes(path: Path) -> bytes:
    """Normalized PNG bytes for `path`. Reads only; writes nothing anywhere."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img) or img
            img = img.convert("RGB")
            if max(img.size) > MAX_IMAGE_PX:
                img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
    except (OSError, ValueError) as e:
        raise IngestError(f"{path}: not a readable image ({e})") from e
    return buffer.getvalue()


def encode_image(path: Path) -> str:
    """Base64 for the API's `images` field."""
    return base64.b64encode(prepare_image_bytes(path)).decode("ascii")


async def extract_from_image(
    client: OllamaClient,
    path: Path,
    id_factory: Callable[[Path, int, str | None], str],
    *,
    seed: int | None = None,
) -> list[Problem]:
    """Every problem the `vision` role can read off `path`.

    Raises StructuredCallError if the model never returns valid JSON; an image
    with no problem on it legitimately yields an empty list.
    """
    result = await client.structured(
        "vision",
        [
            {"role": "system", "content": prompts.load("vision_extract")},
            {"role": "user", "content": prompts.load("vision_extract_user")},
        ],
        VisionExtraction,
        images=[encode_image(path)],
        seed=seed,
    )
    problems: list[Problem] = []
    for extracted in result.value.problems:
        statement = extracted.statement_md.strip()
        if not statement:
            # A blank entry is the model padding its list; a Problem with no
            # statement would fail far downstream with a much worse message.
            log.warning("%s: dropped a vision entry with an empty statement", path)
            continue
        index = len(problems) + 1
        problems.append(
            Problem(
                id=id_factory(path, index, None),
                source=str(path),
                source_index=index,
                statement_md=statement,
                given_answer=extracted.answer_if_shown or None,
                diagram_description=extracted.diagram_description.strip() or None,
                confidence=extracted.confidence,
            )
        )
    return problems
