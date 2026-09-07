from __future__ import annotations

import copy
import random

import pytest

from conftest import PEN_BLUEPRINT
from mathscrambler.blueprint import Blueprint
from mathscrambler.sampler import (
    _int_range,
    answer_is_nice_like,
    check_entity_swap,
    degeneracy,
    draw_parameters,
    math_and_numbers_unchanged,
    sample_variants,
)
from mathscrambler.sandbox import Answer

FOURTEEN = Answer(kind="integer", text="14", numeric=14.0)


def variant_of(**changes) -> Blueprint:
    raw = copy.deepcopy(PEN_BLUEPRINT)
    raw.update(changes)
    return Blueprint.model_validate(raw)


def sample(bp: Blueprint, original=FOURTEEN, **kwargs):
    kwargs.setdefault("n", 3)
    kwargs.setdefault("seed", 42)
    kwargs.setdefault("max_samples", 400)
    return sample_variants(bp, original, **kwargs)


# --------------------------------------------------------------------------- draws


def test_draws_stay_in_the_originals_magnitude_class(pen_blueprint):
    rng = random.Random(7)
    for _ in range(200):
        drawn = draw_parameters(pen_blueprint, rng)
        assert 1 <= drawn["p1"] <= 9  # original 3
        assert 10 <= drawn["p3"] <= 99  # original 21


def test_relaxing_widens_the_range(pen_blueprint):
    rng = random.Random(7)
    widened = {draw_parameters(pen_blueprint, rng, relaxed=True)["p3"] for _ in range(300)}
    assert any(value < 10 for value in widened), "relaxed draws must be able to leave the digit class"


@pytest.mark.parametrize("value", [1, 3, 9, 10, 21, 99, 100, 512, 1000])
def test_relaxing_never_takes_away_a_draw_it_used_to_allow(value: int):
    """`21` has a digit class of 10..99 but a 4x band of only 5..84. If relaxing
    replaced the class instead of widening it, the fallback could exclude draws
    the strict pass had already been making."""
    strict = _int_range(value, relaxed=False)
    relaxed = _int_range(value, relaxed=True)
    assert relaxed[0] <= strict[0] and relaxed[1] >= strict[1]


def test_negative_originals_keep_their_sign():
    bp = variant_of(parameters=[{"slot": "p1", "original": -12, "type": "int"}], constraints=[])
    bp = Blueprint.model_validate(
        {
            **bp.model_dump(),
            "template_md": "A {E1} owes {E2}: ${p1}$.",
            "solver_code": "def solve(p1):\n    return p1 * 2\n",
        }
    )
    rng = random.Random(3)
    assert all(draw_parameters(bp, rng)["p1"] < 0 for _ in range(50))


# --------------------------------------------------------------------------- sampling


def test_produces_variants_that_obey_the_blueprint(pen_blueprint):
    report = sample(pen_blueprint)
    assert report.ok, report.failure
    assert len(report.variants) == 3
    originals = pen_blueprint.original_params()
    for variant in report.variants:
        p = variant.params
        assert p["p3"] % p["p1"] == 0 and p["p3"] > p["p1"] and p["p1"] > 1
        assert variant.answer.text == str(p["p3"] // p["p1"] * p["p2"])
        # Every number changes: a variant sharing a number with its original
        # reads as a typo of it rather than as a new problem.
        assert all(p[slot] != originals[slot] for slot in p)


def test_the_rendered_statement_carries_the_new_numbers(pen_blueprint):
    variant = sample(pen_blueprint, n=1).variants[0]
    assert f"${variant.params['p1']}$" in variant.statement_md
    assert "{p1}" not in variant.statement_md and "{E1}" not in variant.statement_md
    assert "shop" in variant.statement_md  # no entity map given: originals kept


def test_the_same_seed_draws_the_same_variants(pen_blueprint):
    first = sample(pen_blueprint)
    second = sample(pen_blueprint)
    assert [v.params for v in first.variants] == [v.params for v in second.variants]
    assert [v.params for v in sample(pen_blueprint, seed=43).variants] != [v.params for v in first.variants]


def test_each_variant_is_seeded_independently(pen_blueprint):
    """Variant 2 must not shift when variant 1 needs a different number of draws,
    or adding a constraint would silently renumber every later variant."""
    two = sample(pen_blueprint, n=2)
    three = sample(pen_blueprint, n=3)
    assert [v.params for v in three.variants[:2]] == [v.params for v in two.variants]


def test_variants_differ_from_each_other(pen_blueprint):
    report = sample(pen_blueprint, n=5)
    draws = [tuple(v.params.values()) for v in report.variants]
    assert len(set(draws)) == len(draws)


def test_answers_keep_the_shape_the_original_had():
    """A variant whose answer is 47/93 when the original's was 8 has the same
    solution path but is not the same exercise (Section 3, Step C)."""
    bp = variant_of(
        constraints=["p1 > 1", "p3 > p1", "p2 > 0"],  # divisibility dropped on purpose
        solver_code="def solve(p1, p2, p3):\n    from fractions import Fraction\n"
        "    return Fraction(p3, p1) * p2\n",
    )
    report = sample(bp, n=4)
    assert report.ok, report.failure
    assert all(v.answer.kind == "integer" for v in report.variants)
    assert any("answer was rational" in reason for reason in report.rejections)

    loose = sample(bp, n=4, keep_nice_answers=False)
    assert any(v.answer.kind == "rational" for v in loose.variants)


def test_degenerate_answers_are_rejected_unless_the_original_was_degenerate():
    bp = variant_of(
        constraints=["p1 > 1", "p3 > p1", "p2 > 0"],
        solver_code="def solve(p1, p2, p3):\n    return p3 // p3 + p2 - p2\n",  # always 1
    )
    assert not sample(bp, n=1, max_samples=60).ok
    one = Answer(kind="integer", text="1", numeric=1.0)
    assert sample(bp, original=one, n=1, max_samples=60).ok, "the original was 1 too, so 1 is allowed"
    assert sample(bp, n=1, max_samples=60, reject_degenerate=False).ok


def test_an_impossible_constraint_reports_why_instead_of_hanging(pen_blueprint):
    bp = variant_of(constraints=["p1 > 1", "p1 < 1"])
    report = sample(bp, n=1, max_samples=50, relaxed_extra_samples=50)
    assert not report.ok
    assert "could not find variant 1 of 1" in report.failure
    assert "constraints not satisfied" in report.summary()


def test_relaxation_rescues_a_draw_the_strict_range_cannot_make():
    """p1 may be 4 or 12. 4 is the original, so only a relaxed draw can succeed."""
    bp = Blueprint.model_validate(
        {
            **PEN_BLUEPRINT,
            "parameters": [{"slot": "p1", "original": 4, "type": "int", "description": "n"}],
            "constraints": ["p1 == 4 or p1 == 12"],
            "template_md": "A {E1} counts {E2}: ${p1}$.",
            "solver_code": "def solve(p1):\n    return p1 * 3\n",
        }
    )
    report = sample(bp, original=Answer(kind="integer", text="12", numeric=12.0), n=1, max_samples=60)
    assert report.ok, report.failure
    assert report.variants[0].params["p1"] == 12
    assert report.variants[0].relaxed


def test_a_solver_that_always_hangs_gives_up_instead_of_retrying_forever(pen_blueprint):
    bp = variant_of(solver_code="def solve(p1, p2, p3):\n    while True:\n        pass\n")
    report = sample(bp, n=1, max_samples=50, timeout_s=0.3)
    assert not report.ok
    assert any("solver" in reason for reason in report.rejections)


def test_proof_blueprints_are_refused_with_a_reason():
    bp = variant_of(kind="proof", answer_type="proof", solver_code=None)
    report = sample(bp, n=1)
    assert not report.ok and "no solver" in report.failure


def test_entity_maps_are_applied_to_the_statement(pen_blueprint):
    maps = [{"E1": "market stall", "E2": "marbles"}]
    variant = sample(pen_blueprint, n=1, entity_maps=maps).variants[0]
    assert "market stall" in variant.statement_md and "marbles" in variant.statement_md
    assert "pens" not in variant.statement_md


# --------------------------------------------------------------------------- guards


@pytest.mark.parametrize(
    ("original", "replacement", "allowed"),
    [
        ("Alice", "Priya", True),
        ("apples", "marbles", True),
        ("square ABCD", "square PQRS", True),
        ("square ABCD", "rectangle PQRS", False),  # changes the mathematics
        ("the median", "the middle one", False),
        ("a prime number", "a lucky number", False),
        ("apples", "apples", False),
        ("apples", "  ", False),
        ("apples", "7 marbles", False),
    ],
)
def test_the_guard_list_protects_mathematically_loaded_nouns(original, replacement, allowed):
    assert (check_entity_swap(original, replacement) is None) is allowed


def test_the_grammar_pass_may_touch_words_but_not_numbers_or_maths():
    before = "A shop sells 3 pens for $2$ dollars, so $x = 6$."
    assert math_and_numbers_unchanged(before, "The shop sells 3 pens for $2$ dollars, so $x = 6$.")
    assert not math_and_numbers_unchanged(before, "A shop sells 4 pens for $2$ dollars, so $x = 6$.")
    assert not math_and_numbers_unchanged(before, "A shop sells 3 pens for $2$ dollars, so $x = 7$.")


def test_answer_shape_comparisons():
    integer = Answer(kind="integer", text="8", numeric=8.0)
    assert answer_is_nice_like(integer, Answer(kind="integer", text="9", numeric=9.0))
    assert not answer_is_nice_like(integer, Answer(kind="rational", text="9/2", numeric=4.5))
    terminating = Answer(kind="rational", text="3/4", numeric=0.75)
    assert answer_is_nice_like(terminating, Answer(kind="rational", text="1/8", numeric=0.125))
    # 1/3 does not terminate, and the original's did.
    assert not answer_is_nice_like(terminating, Answer(kind="rational", text="1/3", numeric=1 / 3))
    repeating = Answer(kind="rational", text="1/3", numeric=1 / 3)
    assert answer_is_nice_like(repeating, terminating)


def test_degeneracy_names_its_reasons():
    params = {"p1": 3, "p2": 7}
    assert degeneracy(Answer(kind="integer", text="0", numeric=0.0), params) == {"zero"}
    assert degeneracy(Answer(kind="integer", text="1", numeric=1.0), params) == {"one"}
    assert degeneracy(Answer(kind="integer", text="7", numeric=7.0), params) == {"equal to an input"}
    assert degeneracy(Answer(kind="integer", text="12", numeric=12.0), params) == set()
