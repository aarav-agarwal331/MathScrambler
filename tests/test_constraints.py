from __future__ import annotations

import pytest

from mathscrambler.constraints import (
    ConstraintError,
    compile_all,
    compile_constraint,
    failures,
    satisfied,
)

SLOTS = ["p1", "p2", "p3"]


def compiled(source: str):
    return compile_constraint(source, SLOTS)


def test_evaluates_arithmetic_over_slots():
    c = compiled("p1 % p2 == 0 and p1 > p2")
    assert c.evaluate({"p1": 12, "p2": 4})
    assert not c.evaluate({"p1": 13, "p2": 4})


def test_only_the_slots_it_uses_are_required():
    c = compiled("p1 > 0")
    assert c.slots == frozenset({"p1"})
    assert c.evaluate({"p1": 5})  # p2/p3 need not be supplied
    with pytest.raises(ConstraintError, match="no value for p1"):
        c.evaluate({"p2": 1})


@pytest.mark.parametrize(
    "source",
    [
        "p1.__class__",  # attribute access, the usual sandbox escape
        "__import__('os').listdir('.')",
        "open('/etc/passwd')",
        "(lambda: 1)()",
        "[p1 for p1 in ()][0]",  # subscripting
        "p1 if (x := 2) else p2",  # walrus binds a name
    ],
)
def test_expressions_outside_the_allowlist_are_refused(source: str):
    """These run in *this* process, so the AST check is the only thing between a
    model-written constraint and the CLI's own interpreter."""
    with pytest.raises(ConstraintError):
        compiled(source)


def test_exponents_must_be_small_literals():
    assert compiled("p1**2 > p2").evaluate({"p1": 4, "p2": 3})
    # `2 ** p1` with a slot-sized exponent would hang or exhaust memory here.
    with pytest.raises(ConstraintError, match="literal exponent"):
        compiled("2**p1 > p2")
    with pytest.raises(ConstraintError, match="literal exponent"):
        compiled("p1**99 > p2")


def test_unknown_names_are_named_and_the_slots_listed():
    with pytest.raises(ConstraintError, match=r"unknown name 'q7'.*p1, p2, p3"):
        compiled("q7 > p1")


def test_prose_constraints_fail_with_a_pointer_to_the_helpers():
    """The spec's own example writes one constraint in prose. It cannot be
    evaluated, so the error has to be usable as regeneration feedback."""
    with pytest.raises(ConstraintError) as excinfo:
        compiled("triangle inequality on p1,p2,p3")
    message = str(excinfo.value)
    assert "not a Python expression" in message and "triangle" in message


def test_domain_helpers():
    assert compiled("triangle(p1, p2, p3)").evaluate({"p1": 3, "p2": 4, "p3": 5})
    assert not compiled("triangle(p1, p2, p3)").evaluate({"p1": 1, "p2": 2, "p3": 3})  # degenerate
    assert compiled("right_triangle(p1, p2, p3)").evaluate({"p1": 3, "p2": 4, "p3": 5})
    assert compiled("is_prime(p1) and not is_prime(p2)").evaluate({"p1": 17, "p2": 18})
    assert compiled("coprime(p1, p2)").evaluate({"p1": 9, "p2": 14})
    assert compiled("distinct(p1, p2, p3)").evaluate({"p1": 1, "p2": 2, "p3": 3})
    assert not compiled("distinct(p1, p2, p3)").evaluate({"p1": 1, "p2": 1, "p3": 3})
    assert compiled("divides(p1, p2)").evaluate({"p1": 3, "p2": 12})
    assert not compiled("divides(p1, p2)").evaluate({"p1": 0, "p2": 12}), "divides(0, x) must not raise"


def test_factorial_will_not_be_asked_for_an_astronomical_number():
    with pytest.raises(ConstraintError, match="factorial"):
        compiled("factorial(p1) > 0").evaluate({"p1": 5000})


def test_comprehensions_may_bind_their_own_names():
    c = compiled("all(v > 0 for v in (p1, p2))")
    assert c.evaluate({"p1": 1, "p2": 2})
    assert not c.evaluate({"p1": 1, "p2": -2})


def test_a_constraint_must_be_a_condition_not_a_value():
    with pytest.raises(ConstraintError, match="true/false"):
        compiled("(p1, p2)").evaluate({"p1": 1, "p2": 2})


def test_an_evaluation_error_rejects_the_candidate_rather_than_crashing():
    c = compiled("p1 % p2 == 0")
    with pytest.raises(ConstraintError, match="ZeroDivisionError"):
        c.evaluate({"p1": 4, "p2": 0})
    # The sampler draws thousands of candidates; a division by zero in one of
    # them is a rejected draw, not an aborted run.
    assert satisfied([c], {"p1": 4, "p2": 0}) is False


def test_compile_all_collects_every_error_instead_of_stopping_at_the_first():
    good, errors = compile_all(["p1 > 0", "wat?", "q9 > 1", "p2 > 0"], SLOTS)
    assert [c.source for c in good] == ["p1 > 0", "p2 > 0"]
    assert len(errors) == 2, "one regeneration round should be able to fix everything at once"


def test_failures_reports_which_constraints_did_not_hold():
    good, _ = compile_all(["p1 > 100", "p2 > 0", "p3 % p1 == 0"], SLOTS)
    assert failures(good, {"p1": 3, "p2": 5, "p3": 9}) == ["p1 > 100"]


def test_empty_constraints_are_rejected_not_ignored():
    with pytest.raises(ConstraintError, match="empty"):
        compiled("   ")


def test_string_literals_are_refused_so_a_constraint_cannot_allocate_gigabytes():
    """Constraints are evaluated in the CLI's own process. `"a" * 10**8` is a
    legal expression over an allowed operator, and there is no sandbox here."""
    with pytest.raises(ConstraintError, match="only numeric literals"):
        compiled("len('a' * 10**8) > p1")
    assert compiled("p1 > 1000").evaluate({"p1": 2000})  # ordinary literals still fine
