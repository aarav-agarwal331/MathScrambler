from __future__ import annotations

import copy

import pytest

from conftest import PEN_BLUEPRINT
from mathscrambler.blueprint import (
    Blueprint,
    answer_matches_given,
    check_solver,
    render_template,
    validate_blueprint,
)
from mathscrambler.sandbox import Answer


def variant_of(**changes) -> Blueprint:
    raw = copy.deepcopy(PEN_BLUEPRINT)
    raw.update(changes)
    return Blueprint.model_validate(raw)


def test_a_good_blueprint_validates_clean(pen_blueprint):
    assert validate_blueprint(pen_blueprint) == []


def test_every_declared_slot_must_appear_in_the_template():
    bp = variant_of(template_md="A {E1} sells {E2} at ${p1}$ for $\\${p2}$. What is the cost?")
    assert any("{p3}" in problem for problem in validate_blueprint(bp))


def test_a_template_may_not_reference_an_undeclared_slot():
    bp = variant_of(template_md=PEN_BLUEPRINT["template_md"] + " Also {p9}.")
    assert any("{p9}" in p and "not a declared slot" in p for p in validate_blueprint(bp))


def test_latex_braces_are_not_mistaken_for_slots():
    """`\\frac{1}{2}` and `\\sqrt{x}` are ordinary LaTeX. A looser placeholder
    pattern would read them as undeclared slots and reject every real problem."""
    bp = variant_of(
        template_md=PEN_BLUEPRINT["template_md"] + r" Give the answer as $\frac{p}{q}$ or $\sqrt{x}$."
    )
    assert validate_blueprint(bp) == []
    rendered = render_template(bp.template_md, {"E1": "shop", "E2": "pens", "p1": 3, "p2": 2, "p3": 21})
    assert r"\frac{p}{q}" in rendered and r"\sqrt{x}" in rendered


def test_slot_names_must_follow_the_scheme():
    bp = variant_of(
        parameters=[{"slot": "price", "original": 3, "type": "int"}, *PEN_BLUEPRINT["parameters"][1:]],
        constraints=[],
        template_md="A {E1} sells {E2} at {price} for {p2}, so {p3}.",
    )
    assert any("must look like p1" in p for p in validate_blueprint(bp))


def test_duplicate_slots_are_reported():
    duplicated = [*PEN_BLUEPRINT["parameters"], {"slot": "p1", "original": 9, "type": "int"}]
    assert any("duplicate slot" in p for p in validate_blueprint(variant_of(parameters=duplicated)))


def test_solver_signature_must_match_the_parameter_slots():
    problems = validate_blueprint(variant_of(solver_code="def solve(p1, p2):\n    return p1\n"))
    assert any("missing arguments for p3" in p for p in problems)
    problems = validate_blueprint(variant_of(solver_code="def solve(p1, p2, p3, p4):\n    return p1\n"))
    assert any("p4" in p and "not parameter slots" in p for p in problems)
    problems = validate_blueprint(variant_of(solver_code="def solve(**kw):\n    return 1\n"))
    assert any("*args/**kwargs" in p for p in problems)
    problems = validate_blueprint(variant_of(solver_code="def answer(p1, p2, p3):\n    return 1\n"))
    assert any("no function named solve" in p for p in problems)


def test_an_unparseable_solver_is_reported_not_executed():
    problems = validate_blueprint(variant_of(solver_code="def solve(p1, p2, p3:\n"))
    assert any("does not parse" in p for p in problems)


def test_proofs_have_no_solver_and_solvers_are_not_proofs():
    assert any("must go together" in p for p in validate_blueprint(variant_of(answer_type="proof")))
    proof = variant_of(kind="proof", answer_type="proof", solver_code=None)
    assert validate_blueprint(proof) == []
    assert any("needs answer_type" in p for p in validate_blueprint(variant_of(kind="proof")))


def test_invalid_constraints_surface_as_blueprint_problems():
    problems = validate_blueprint(variant_of(constraints=["p1 % p2 == 0", "triangle inequality on p1,p2"]))
    assert any("not a Python expression" in p for p in problems)


def test_a_parameter_slot_may_not_shadow_a_constraint_helper():
    # There is no legal `sum` slot, but the check exists because a slot that
    # shadowed a helper would silently change what every constraint means.
    bp = variant_of(
        parameters=[{"slot": "min", "original": 3, "type": "int"}, *PEN_BLUEPRINT["parameters"][1:]],
        constraints=[],
        template_md="A {E1} sells {E2}: {min} {p2} {p3}.",
    )
    assert any("shadows a constraint helper" in p for p in validate_blueprint(bp))


def test_solution_outline_may_not_be_empty():
    assert any("solution_outline" in p for p in validate_blueprint(variant_of(solution_outline=[])))


# --------------------------------------------------------------------- the solver gate


def test_the_solver_must_reproduce_the_printed_answer(pen_blueprint):
    check = check_solver(pen_blueprint, given_answer="Ans: $14")
    assert check.ok, check.problems
    assert check.answer.text == "14"


def test_a_solver_that_computes_something_else_fails_the_gate():
    bp = variant_of(solver_code="def solve(p1, p2, p3):\n    return p1 + p2 + p3\n")
    check = check_solver(bp, given_answer="$14")
    assert not check.ok
    assert "does not reproduce the original problem" in check.feedback


def test_original_parameters_must_satisfy_the_blueprints_own_constraints():
    # 21 is not divisible by 4: the blueprint contradicts the problem it came from.
    bp = variant_of(
        parameters=[{"slot": "p1", "original": 4, "type": "int"}, *PEN_BLUEPRINT["parameters"][1:]]
    )
    check = check_solver(bp)
    assert not check.ok
    assert "do not satisfy the blueprint's own constraints" in check.feedback


def test_a_solver_that_crashes_on_the_original_fails_the_gate():
    bp = variant_of(solver_code="def solve(p1, p2, p3):\n    return p3 // (p1 - p1)\n")
    check = check_solver(bp)
    assert not check.ok and "ZeroDivisionError" in check.feedback


def test_a_solver_that_will_not_load_fails_the_gate_without_raising():
    bp = variant_of(solver_code="import socket\ndef solve(p1, p2, p3):\n    return 1\n")
    check = check_solver(bp)
    assert not check.ok and "could not be loaded" in check.feedback


def test_a_hanging_solver_is_killed_and_reported():
    bp = variant_of(solver_code="def solve(p1, p2, p3):\n    while True:\n        pass\n")
    check = check_solver(bp, timeout_s=1.0)
    assert not check.ok and "timeout" in check.feedback


def test_proof_blueprints_pass_the_gate_with_nothing_to_run():
    assert check_solver(variant_of(kind="proof", answer_type="proof", solver_code=None)).ok


@pytest.mark.parametrize(
    ("text", "given", "expected"),
    [
        ("14", "$14", True),
        ("14", "Ans: $14.00", True),
        ("14", "14 dollars", True),
        ("1400", "$1,400", True),
        ("14", "x = 14", True),
        ("3/4", "3/4 of a litre", True),
        ("3/4", "0.75", True),
        ("14", "$41", False),
        ("14", "", False),
    ],
)
def test_answer_matching_is_lenient_about_how_the_source_wrote_it(text, given, expected):
    """`given_answer` is whatever the book printed. Rejecting a correct solver
    over a currency symbol would send a good blueprint back for regeneration."""
    kind = "rational" if "/" in text else "integer"
    numeric = float(text.split("/")[0]) / float(text.split("/")[1]) if "/" in text else float(text)
    assert answer_matches_given(Answer(kind=kind, text=text, numeric=numeric), given) is expected


def test_a_number_is_not_matched_by_being_a_substring_of_another_number():
    """"4" appears inside "$41". Matching on substrings would let a solver that
    computes 4 look as though it reproduced a printed answer of 41."""
    assert not answer_matches_given(Answer(kind="integer", text="4", numeric=4.0), "$41")
    assert not answer_matches_given(Answer(kind="integer", text="2", numeric=2.0), "x = 12")
    # Expressions have no numeric value, so substring matching is all they have.
    assert answer_matches_given(Answer(kind="expression", text="2*x + 3"), "the answer is 2*x + 3")
