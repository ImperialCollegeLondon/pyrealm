"""Tests the functions in bounds_checker.py."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="lower,upper,interval_type,raises,msg",
    argvalues=(
        pytest.param(0, 1, "[]", does_not_raise(), None, id="good"),
        pytest.param(
            1,
            1,
            "[]",
            pytest.raises(ValueError),
            "Bounds equal or reversed: 1, 1",
            id="lower_equal",
        ),
        pytest.param(
            2,
            1,
            "[]",
            pytest.raises(ValueError),
            "Bounds equal or reversed: 2, 1",
            id="lower_greater",
        ),
        pytest.param(
            0,
            1,
            "][",
            pytest.raises(ValueError),
            "Unknown interval type: ][",
            id="bad_interval_type",
        ),
    ),
)
def test_Bounds(lower, upper, interval_type, raises, msg):
    """Test the Bounds data class __post_init__ validation."""
    from pyrealm.core.bounds import Bounds

    with raises as excep:
        _ = Bounds(
            var_name="test",
            lower=lower,
            upper=upper,
            interval_type=interval_type,
            unit="-",
        )

    if excep is not None:
        assert str(excep.value) == msg


@pytest.mark.parametrize(
    argnames="var_name, interval_arg, values, warns",
    argvalues=[
        pytest.param("test", "[]", np.array([1.5, 3.5]), False, id="within_[]"),
        pytest.param("test", "()", np.array([1.5, 3.5]), False, id="within_()"),
        pytest.param("test", "(]", np.array([1.5, 3.5]), False, id="within_(]"),
        pytest.param("test", "[)", np.array([1.5, 3.5]), False, id="within_[)"),
        pytest.param("test", "[]", np.array([0.5, 3.5]), True, id="below_[]"),
        pytest.param("test", "()", np.array([0.5, 3.5]), True, id="below_()"),
        pytest.param("test", "(]", np.array([0.5, 3.5]), True, id="below_(]"),
        pytest.param("test", "[)", np.array([0.5, 3.5]), True, id="below_[)"),
        pytest.param("test", "[]", np.array([1.5, 4.5]), True, id="above_[]"),
        pytest.param("test", "()", np.array([1.5, 4.5]), True, id="above_()"),
        pytest.param("test", "(]", np.array([1.5, 4.5]), True, id="above_(]"),
        pytest.param("test", "[)", np.array([1.5, 4.5]), True, id="above_[)"),
        pytest.param("test", "[]", np.array([1, 3]), False, id="at_lower_[]"),
        pytest.param("test", "[]", np.array([2, 4]), False, id="at_upper[]"),
        pytest.param("test", "()", np.array([1, 3]), True, id="at_lower_()"),
        pytest.param("test", "()", np.array([2, 4]), True, id="at_upper_()"),
        pytest.param("test", "(]", np.array([1, 3]), True, id="at_lower_(]"),
        pytest.param("test", "(]", np.array([2, 4]), False, id="at_upper_(]"),
        pytest.param("test", "[)", np.array([1, 3]), False, id="at_lower_[)"),
        pytest.param("test", "[)", np.array([2, 4]), True, id="at_upper_[)"),
        pytest.param("unknown", "[)", np.array([2, 4]), True, id="no_var_bounds"),
    ],
)
def test_BoundsChecker(var_name, interval_arg, values, warns):
    """The the BoundsChecker utility raises warnings as expected."""
    from pyrealm.core.bounds import Bounds, BoundsChecker

    bounds_checker = BoundsChecker()
    bounds_checker.update(Bounds("test", 1, 4, interval_arg, "-"))

    context = pytest.warns(UserWarning) if warns else does_not_raise()

    with context as warning:
        _ = bounds_checker.check(var_name, values)

    if warning is not None:
        exp_msg = (
            "Variable 'test' (-) contains values outside"
            if var_name == "test"
            else "Variable 'unknown' is not configured"
        )
        assert str(warning.list[0].message).startswith(exp_msg)
