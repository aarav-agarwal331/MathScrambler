"""Ingest unit tests. No model, no network — the vision path runs against the HTTP stub."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from mathscrambler import ingest, paths, prompts
from mathscrambler.config import ProfileRoles, RoleConfig
from mathscrambler.ingest import IdFactory, IngestError, discover, load_text_inputs
from mathscrambler.ingest.images import MAX_IMAGE_PX, encode_image, extract_from_image, prepare_image_bytes
from mathscrambler.ingest.models import LOW_CONFIDENCE, Problem
from mathscrambler.ingest.text import parse_directives, split_chunks
from mathscrambler.ollama_client import OllamaClient

EXAMPLES = paths.repo_root() / "examples"

VISION = RoleConfig(tag="vision-test:v", num_ctx=8192, temperature=0.0, think=False)
ROLES = ProfileRoles(vision=VISION, reasoner=RoleConfig(tag="reasoner-test"), fast="vision")


def _chat_body(content: str) -> dict:
    return {
        "model": "vision-test:v",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "total_duration": 1_000_000_000,
        "eval_count": 20,
        "eval_duration": 1_000_000_000,
    }


def _vision_stub(http_stub, extraction: dict) -> OllamaClient:
    http_stub.post_route("/api/chat", _chat_body(json.dumps(extraction)))
    return OllamaClient(f"http://127.0.0.1:{http_stub.port}", ROLES)


def _png(path: Path, size: tuple[int, int] = (64, 48), color: str = "white") -> Path:
    Image.new("RGB", size, color).save(path)
    return path


# --------------------------------------------------------------------------- splitting


def test_split_chunks_on_dash_lines():
    assert split_chunks("one\n---\ntwo\n----\nthree") == ["one", "two", "three"]


def test_split_chunks_ignores_separators_inside_code_fences():
    """A `---` inside fenced code is content; splitting there would corrupt the problem."""
    text = "before\n```\n---\nstill code\n```\nafter\n---\nsecond"
    first, second = split_chunks(text)
    assert "still code" in first and "before" in first and "after" in first
    assert second == "second"


def test_split_chunks_drops_empty_and_whitespace_chunks():
    assert split_chunks("---\n\na\n---\n   \n---\nb\n---\n") == ["a", "b"]


# --------------------------------------------------------------------------- directives


def test_directives_parse_in_both_comment_styles():
    statement, found = parse_directives(
        "Find $x$.\n<!-- answer: 8 -->\n% tags: algebra, linear\n<!-- kind: proof -->"
    )
    assert statement == "Find $x$."
    assert found == {"answer": "8", "tags": "algebra, linear", "kind": "proof"}


def test_unrecognized_comment_stays_in_the_statement():
    """Silently deleting an author's text is worse than ignoring a key we don't know."""
    statement, found = parse_directives("Solve it.\n<!-- note: from the 1998 paper -->")
    assert "<!-- note: from the 1998 paper -->" in statement
    assert found == {}


def test_bad_kind_directive_names_the_problem_it_came_from(tmp_path: Path):
    bad = tmp_path / "x.md"
    bad.write_text("Fine.\n---\nSolve it.\n<!-- kind: puzzle -->")
    result = load_text_inputs([bad])
    assert result.problems == []  # the file is rejected whole; nothing half-loaded
    assert "problem 2: kind must be one of" in result.skipped[0].reason


# --------------------------------------------------------------------------- text loaders


def test_markdown_carries_metadata_and_indices(tmp_path: Path):
    src = tmp_path / "algebra.md"
    src.write_text(
        "First problem.\n<!-- answer: 12 -->\n<!-- tags: algebra, sums -->\n"
        "---\n"
        "Second problem.\n<!-- kind: proof -->\n"
    )
    problems = load_text_inputs([src]).problems
    assert [p.id for p in problems] == ["algebra-01", "algebra-02"]
    assert [p.source_index for p in problems] == [1, 2]
    assert problems[0].given_answer == "12"
    assert problems[0].tags == ["algebra", "sums"]
    assert problems[0].kind_hint is None
    assert problems[1].kind_hint == "proof"
    assert problems[1].statement_md == "Second problem."
    assert problems[0].source == str(src)


def test_txt_follows_the_same_rules(tmp_path: Path):
    src = tmp_path / "rates.txt"
    src.write_text("A tap fills a tank.\n---\nAnother tap.\n")
    assert len(load_text_inputs([src]).problems) == 2


def test_tex_reads_problem_environments(tmp_path: Path):
    src = tmp_path / "g.tex"
    src.write_text(
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
        "\\begin{problem}\nFind the area of $T$.\n% answer: 60\n\\end{problem}\n"
        "\\begin{exercise}\nFind the perimeter.\n\\end{exercise}\n"
        "\\end{document}\n"
    )
    problems = load_text_inputs([src]).problems
    assert [p.statement_md for p in problems] == ["Find the area of $T$.", "Find the perimeter."]
    assert problems[0].given_answer == "60"
    # the preamble is not a problem
    assert all("documentclass" not in p.statement_md for p in problems)


def test_tex_without_environments_splits_on_dashes_inside_the_document(tmp_path: Path):
    src = tmp_path / "g.tex"
    src.write_text("\\documentclass{article}\n\\begin{document}\nOne.\n---\nTwo.\n\\end{document}\n")
    assert [p.statement_md for p in load_text_inputs([src]).problems] == ["One.", "Two."]


def test_json_accepts_objects_and_bare_strings(tmp_path: Path):
    src = tmp_path / "c.json"
    src.write_text(
        json.dumps(
            {
                "problems": [
                    {"id": "committee", "statement_md": "How many committees?", "answer": 105,
                     "tags": ["counting"], "kind": "computational"},
                    "A bare string problem.",
                ]
            }
        )
    )
    problems = load_text_inputs([src]).problems
    assert problems[0].id == "committee"
    assert problems[0].given_answer == "105"  # numbers are normalized to text
    assert problems[0].tags == ["counting"]
    assert problems[1].id == "c-02"
    assert problems[1].statement_md == "A bare string problem."


def test_json_top_level_list_is_accepted(tmp_path: Path):
    src = tmp_path / "c.json"
    src.write_text(json.dumps([{"statement": "One."}, {"text": "Two."}]))
    assert len(load_text_inputs([src]).problems) == 2


def test_json_errors_are_reported_not_raised(tmp_path: Path):
    """One malformed input must not take the whole run down with it."""
    good = tmp_path / "good.md"
    good.write_text("Fine problem.")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"tags": ["no statement here"]}]))
    result = load_text_inputs([good, bad])
    assert [p.statement_md for p in result.problems] == ["Fine problem."]
    assert len(result.skipped) == 1
    assert "no statement" in result.skipped[0].reason
    assert result.skipped[0].path == bad


def test_invalid_json_reports_the_line(tmp_path: Path):
    src = tmp_path / "b.json"
    src.write_text("{not json")
    assert "invalid JSON at line 1" in load_text_inputs([src]).skipped[0].reason


# --------------------------------------------------------------------------- discovery


def test_discover_walks_directories_and_skips_furniture(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub" / "c.json").write_text("[]")
    (tmp_path / "sub" / "page.png").write_bytes(b"")
    (tmp_path / "README.md").write_text("not a problem")
    (tmp_path / ".hidden.md").write_text("hidden")
    (tmp_path / "notes.pdf").write_bytes(b"")
    found = discover([tmp_path])
    assert [p.name for p in found.text_files] == ["a.txt", "b.md", "c.json"]
    assert [p.name for p in found.image_files] == ["page.png"]
    # a directory contributes only problem-shaped files, silently
    assert found.skipped == []


def test_discover_reports_explicit_bad_paths(tmp_path: Path):
    (tmp_path / "notes.pdf").write_bytes(b"")
    found = discover([tmp_path / "notes.pdf", tmp_path / "gone.md"])
    reasons = {s.path.name: s.reason for s in found.skipped}
    assert "unsupported file type .pdf" in reasons["notes.pdf"]
    assert reasons["gone.md"] == "does not exist"
    assert found.files == []


def test_discover_dedupes_a_file_named_twice(tmp_path: Path):
    src = tmp_path / "a.md"
    src.write_text("x")
    assert len(discover([src, tmp_path, src]).files) == 1


# --------------------------------------------------------------------------- ids


def test_id_factory_slugs_and_avoids_collisions():
    ids = IdFactory()
    assert ids(Path("a/Chapter 4 Notes.md"), 1, None) == "chapter-4-notes-01"
    assert ids(Path("b/Chapter 4 Notes.md"), 1, None) == "chapter-4-notes-01-2"
    assert ids(Path("c.md"), 1, "My Problem!") == "my-problem"
    assert ids(Path("d.md"), 1, "My Problem!") == "my-problem-2"


# --------------------------------------------------------------------------- the examples/ set


def test_examples_text_problems_load_with_expected_content():
    result = load_text_inputs([EXAMPLES])
    by_id = {p.id: p for p in result.problems}
    assert set(by_id) == {
        "algebra-01", "algebra-02", "committee-two-women",
        "geometry-01", "proofs-01", "proofs-02", "rates-01",
    }
    assert by_id["algebra-01"].given_answer == "130"
    assert by_id["rates-01"].given_answer.startswith("36/5")
    assert by_id["geometry-01"].tags == ["geometry", "triangle", "area"]
    assert [p.id for p in result.problems if p.kind_hint == "proof"] == ["proofs-01", "proofs-02"]
    # images are present in examples/ but need the vision model
    assert {s.path.name for s in result.skipped} == {
        "page-linear.png", "page-triangle.png", "page-worksheet.png"
    }
    assert all(p.confidence is None for p in result.problems)  # authored, not extracted


def test_examples_directives_never_leak_into_statements():
    for problem in load_text_inputs([EXAMPLES]).problems:
        assert "<!--" not in problem.statement_md
        assert "% answer" not in problem.statement_md
        assert "\\begin{document}" not in problem.statement_md


def test_ingest_never_mutates_its_inputs():
    """Section 1.2: inputs are read-only. Hash the whole tree either side of a load."""
    def digest() -> dict[str, tuple[str, int]]:
        return {
            str(p.relative_to(EXAMPLES)): (
                hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns
            )
            for p in sorted(EXAMPLES.rglob("*"))
            if p.is_file()
        }

    before = digest()
    load_text_inputs([EXAMPLES])
    for path in sorted(EXAMPLES.rglob("*.png")):
        encode_image(path)  # the image path normalizes in memory only
    assert digest() == before
    assert before, "examples/ should not be empty"


# --------------------------------------------------------------------------- images


def test_prepare_image_downscales_and_normalizes(tmp_path: Path):
    big = _png(tmp_path / "big.png", size=(4000, 3000))
    with Image.open(io.BytesIO(prepare_image_bytes(big))) as out:
        assert out.size == (MAX_IMAGE_PX, 1200)  # aspect preserved, long side capped
        assert out.mode == "RGB"


def test_prepare_image_applies_exif_rotation(tmp_path: Path):
    """A phone photo of a page is unreadable to the model if EXIF rotation is ignored."""
    src = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[274] = 6  # orientation: rotate 90° CW on display
    Image.new("RGB", (40, 20), "white").save(src, exif=exif)
    with Image.open(io.BytesIO(prepare_image_bytes(src))) as out:
        assert out.size == (20, 40)


def test_unreadable_image_is_an_ingest_error(tmp_path: Path):
    junk = tmp_path / "fake.png"
    junk.write_bytes(b"not a png at all")
    with pytest.raises(IngestError, match="not a readable image"):
        prepare_image_bytes(junk)


async def test_extract_from_image_builds_problems(tmp_path: Path, http_stub, tmp_home):
    client = _vision_stub(
        http_stub,
        {
            "problems": [
                {"statement_md": "Solve $5(x-3)=2x+9$.", "diagram_description": "",
                 "answer_if_shown": None, "confidence": 0.95},
                {"statement_md": "Find $AC$.", "diagram_description": "right triangle, legs 9 and 12",
                 "answer_if_shown": "15 cm", "confidence": 0.55},
            ]
        },
    )
    async with client:
        problems = await extract_from_image(client, _png(tmp_path / "page-linear.png"), IdFactory())

    assert [p.id for p in problems] == ["page-linear-01", "page-linear-02"]
    assert problems[0].diagram_description is None  # "" normalizes to absent
    assert problems[0].needs_review is False
    assert problems[1].given_answer == "15 cm"
    assert problems[1].diagram_description == "right triangle, legs 9 and 12"
    assert problems[1].needs_review is True  # 0.55 < LOW_CONFIDENCE

    _, payload = http_stub.posts[0]
    assert payload["model"] == "vision-test:v"
    assert payload["options"]["num_ctx"] == 8192
    assert len(payload["messages"][-1]["images"]) == 1  # the image rides on the user turn
    assert payload["format"]["properties"]["problems"]  # schema-constrained decoding


async def test_extract_drops_entries_with_no_statement(tmp_path: Path, http_stub, tmp_home):
    client = _vision_stub(
        http_stub,
        {"problems": [
            {"statement_md": "   ", "confidence": 0.9},
            {"statement_md": "Real one.", "confidence": 0.9},
        ]},
    )
    async with client:
        problems = await extract_from_image(client, _png(tmp_path / "p.png"), IdFactory())
    assert [p.statement_md for p in problems] == ["Real one."]
    assert problems[0].source_index == 1  # indices renumber over kept entries


async def test_ingest_without_a_client_skips_images_but_keeps_text(tmp_path: Path):
    (tmp_path / "a.md").write_text("A text problem.")
    _png(tmp_path / "p.png")
    result = await ingest.ingest([tmp_path])
    assert [p.statement_md for p in result.problems] == ["A text problem."]
    assert "no vision client" in result.skipped[0].reason


async def test_ingest_mixes_text_and_images_in_discovery_order(tmp_path: Path, http_stub, tmp_home):
    (tmp_path / "a.md").write_text("Text problem.")
    _png(tmp_path / "z.png")
    client = _vision_stub(http_stub, {"problems": [{"statement_md": "Image problem.", "confidence": 0.9}]})
    async with client:
        result = await ingest.ingest([tmp_path], client)
    assert [p.statement_md for p in result.problems] == ["Text problem.", "Image problem."]
    assert result.needs_review == []
    assert result.skipped == []


async def test_a_failing_image_is_reported_not_fatal(tmp_path: Path, http_stub, tmp_home):
    (tmp_path / "a.md").write_text("Text problem.")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    client = _vision_stub(http_stub, {"problems": []})
    async with client:
        result = await ingest.ingest([tmp_path], client)
    assert [p.statement_md for p in result.problems] == ["Text problem."]
    assert "not a readable image" in result.skipped[0].reason


def test_low_confidence_boundary_is_exclusive():
    def at(confidence: float) -> Problem:
        return Problem(id="x", source="s", statement_md="q", confidence=confidence)

    assert at(LOW_CONFIDENCE).needs_review is False
    assert at(LOW_CONFIDENCE - 0.01).needs_review is True
    assert Problem(id="x", source="s", statement_md="q").needs_review is False


# --------------------------------------------------------------------------- prompts


def test_vision_prompt_files_exist_and_say_what_matters():
    system = prompts.load("vision_extract")
    assert "LaTeX" in system and "confidence" in system
    assert prompts.load("vision_extract_user")


def test_missing_prompt_is_a_clear_error():
    with pytest.raises(prompts.PromptError, match="no prompt file"):
        prompts.load("no_such_prompt")
