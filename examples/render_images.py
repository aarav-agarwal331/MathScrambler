"""Regenerate the three example page images.

The PNGs under ``examples/images/`` are committed so the test suite and the
vision check have fixed inputs, but they are generated, not mystery binaries:
run ``uv run python examples/render_images.py`` to rebuild them byte-for-byte.

They deliberately look like textbook pages rather than clean prompts — running
headers, page numbers, problem numbering, and one page holding two problems —
because that is what the `vision` role has to cope with, and every one of those
is something the extraction prompt must learn to ignore or split on.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
SERIF = FONT_DIR / "Times New Roman.ttf"
SERIF_BOLD = FONT_DIR / "Times New Roman Bold.ttf"
SERIF_ITALIC = FONT_DIR / "Times New Roman Italic.ttf"

PAGE = (1000, 1360)
PAPER = (252, 251, 246)
INK = (26, 26, 26)
FAINT = (128, 128, 128)
MARGIN = 92
WRAP_COLUMNS = 62


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not path.is_file():
        raise SystemExit(f"missing system font {path} — adjust FONT_DIR for this machine")
    return ImageFont.truetype(str(path), size)


def _new_page(header: str, page_number: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", PAGE, PAPER)
    draw = ImageDraw.Draw(img)
    draw.text((MARGIN, 46), header, font=_font(SERIF_ITALIC, 26), fill=FAINT)
    draw.text((PAGE[0] - MARGIN - 30, 46), str(page_number), font=_font(SERIF_ITALIC, 26), fill=FAINT)
    draw.line([(MARGIN, 84), (PAGE[0] - MARGIN, 84)], fill=FAINT, width=1)
    return img, draw


def _problem(draw: ImageDraw.ImageDraw, y: int, number: str, text: str) -> int:
    """Draw a numbered problem block; return the y below it."""
    draw.text((MARGIN, y), number, font=_font(SERIF_BOLD, 31), fill=INK)
    body = _font(SERIF, 31)
    for line in textwrap.wrap(" ".join(text.split()), width=WRAP_COLUMNS):
        draw.text((MARGIN + 62, y), line, font=body, fill=INK)
        y += 44
    return y


def render_linear(path: Path) -> None:
    img, draw = _new_page("Chapter 4  ·  Linear Equations", 77)
    y = _problem(
        draw,
        150,
        "12.",
        "Solve for x:  5(x - 3) = 2x + 9.",
    )
    _problem(
        draw,
        y + 40,
        "13.",
        "A number is multiplied by 4 and then 7 is subtracted from the result. "
        "The answer is 45. What was the original number?",
    )
    img.save(path)


def render_triangle(path: Path) -> None:
    """A page whose problem is unanswerable without reading the figure."""
    img, draw = _new_page("Chapter 9  ·  Right Triangles", 214)
    y = _problem(
        draw,
        150,
        "7.",
        "In the right triangle below, the right angle is at B. "
        "Find the length of the hypotenuse AC.",
    )
    # Right triangle: B bottom-left, A above it, C to its right. 20 px per cm.
    bx, by = MARGIN + 190, y + 300
    ax, ay = bx, by - 180  # AB = 9 cm
    cx, cy = bx + 240, by  # BC = 12 cm
    draw.line([(ax, ay), (bx, by), (cx, cy), (ax, ay)], fill=INK, width=3)
    draw.line([(bx, by - 26), (bx + 26, by - 26), (bx + 26, by)], fill=INK, width=2)  # right angle
    label = _font(SERIF, 29)
    draw.text((ax - 34, ay - 16), "A", font=label, fill=INK)
    draw.text((bx - 34, by - 4), "B", font=label, fill=INK)
    draw.text((cx + 12, cy - 4), "C", font=label, fill=INK)
    draw.text((bx - 84, (ay + by) // 2 - 16), "9 cm", font=label, fill=INK)
    draw.text(((bx + cx) // 2 - 34, by + 16), "12 cm", font=label, fill=INK)
    img.save(path)


def render_worksheet(path: Path) -> None:
    """Two problems on one page — the split case — one with its answer printed."""
    img, draw = _new_page("Review Worksheet  ·  Ratio and Averages", 3)
    y = _problem(
        draw,
        150,
        "1.",
        "A shop sells pens at 3 for $2. At the same rate, how much do 21 pens cost?",
    )
    draw.text((MARGIN + 62, y + 8), "Ans:  $14", font=_font(SERIF_BOLD, 29), fill=INK)
    _problem(
        draw,
        y + 84,
        "2.",
        "Find the mean of the five numbers 4, 9, 11, 16 and 20.",
    )
    img.save(path)


def main() -> None:
    out = Path(__file__).resolve().parent / "images"
    out.mkdir(exist_ok=True)
    render_linear(out / "page-linear.png")
    render_triangle(out / "page-triangle.png")
    render_worksheet(out / "page-worksheet.png")
    for png in sorted(out.glob("*.png")):
        print(f"wrote {png.relative_to(out.parent.parent)} ({png.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
