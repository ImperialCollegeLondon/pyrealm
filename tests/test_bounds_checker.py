"""Tests the functions in bounds_checker.py."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="input, raises, lowfn, hifun",
    argvalues=[
        ("[]", does_not_raise(), np.less, np.greater),
        ("()", does_not_raise(), np.less_equal, np.greater_equal),
        ("[)", does_not_raise(), np.less, np.greater_equal),
        ("(]", does_not_raise(), np.less_equal, np.greater),
        ("{}", pytest.raises(ValueError), None, None),
    ],
)
def test__get_interval_functions(input, raises, lowfn, hifun):
    """Tests _get_interval_functions"""

    from pyrealm.bounds_checker import _get_interval_functions

    with raises as err:
        low, hi = _get_interval_functions(input)

    if not err:
        assert low == lowfn
        assert hi == hifun


@pytest.mark.parametrize(
    argnames="input, interval_args, context",
    argvalues=[
        (3, (1, 4, "()"), does_not_raise()),
        (np.array([1, 2, 3, 4]), (1.0, 4.0, "()"), pytest.warns(UserWarning)),
        (np.array([1, 2, 3, 4]), (0.9, 4.1, "()"), does_not_raise()),
        (np.array([1, 2, 3, 4]), (1.1, 3.9, "[]"), pytest.warns(UserWarning)),
        (np.array([1, 2, 3, 4]), (1.0, 4.0, "[]"), does_not_raise()),
        (np.array([1, 2, 3, 4]), (1.0, 4.0, "[)"), pytest.warns(UserWarning)),
        (np.array([1, 2, 3, 4]), (1.0, 4.1, "[)"), does_not_raise()),
        (np.array([1, 2, 3, 4]), (1.0, 4.0, "(]"), pytest.warns(UserWarning)),
        (np.array([1, 2, 3, 4]), (0.9, 4.0, "(]"), does_not_raise()),
    ],
)
def test_bounds_checker(input, interval_args, context):

    from pyrealm.bounds_checker import bounds_checker

    with context:
        out = bounds_checker(input, *interval_args)


@pytest.mark.parametrize(
    argnames="input, interval_args, context, exp",
    argvalues=[
        (
            3,
            (1, 4, "()"),
            pytest.raises(TypeError),
            None,
        ),
        (
            np.array([1, 2, 3, 4]),
            (1, 4, "()"),
            pytest.warns(RuntimeWarning),
            np.array([np.nan, 2, 3, np.nan]),
        ),
        (
            np.array([1, 2, 3, 4]),
            (0.9, 4.1, "()"),
            does_not_raise(),
            np.array([1, 2, 3, 4]),
        ),
        (
            np.array([1, 2, 3, 4]),
            (1.1, 3.9, "[]"),
            pytest.warns(RuntimeWarning),
            np.array([np.nan, 2, 3, np.nan]),
        ),
        (
            np.array([1, 2, 3, 4]),
            (1, 4, "[]"),
            does_not_raise(),
            np.array([1, 2, 3, 4]),
        ),
        (
            np.array([1, 2, 3, 4]),
            (1, 4, "[)"),
            pytest.warns(RuntimeWarning),
            np.array([1, 2, 3, np.nan]),
        ),
        (
            np.array([1, 2, 3, 4]),
            (1, 4.1, "[)"),
            does_not_raise(),
            np.array([1, 2, 3, 4]),
        ),
        (
            np.array([1, 2, 3, 4]),
            (1, 4, "(]"),
            pytest.warns(RuntimeWarning),
            np.array([np.nan, 2, 3, 4]),
        ),
        (
            np.array([1, 2, 3, 4]),
            (0.9, 4, "(]"),
            does_not_raise(),
            np.array([1, 2, 3, 4]),
        ),
    ],
)
def test_input_mask(input, interval_args, context, exp):

    from pyrealm.bounds_checker import bounds_mask

    with context:
        out = bounds_mask(input, *interval_args)

        assert np.allclose(out, exp, equal_nan=True)