"""Test the methods used in the FaparLimitation class."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from pyrealm.phenology.fapar_limitation import check_datetimes, get_annual


@pytest.mark.parametrize(
    argnames="datetimes, raises",
    argvalues=[
        (
            np.arange(
                np.datetime64("2010-02-01T00:00"),
                np.datetime64("2011-01-31T23:30"),
                np.timedelta64(30, "m"),
            ),
            does_not_raise(),
        ),
        (
            np.arange(
                np.datetime64("2010-02-01T00:00"),
                np.datetime64("2010-12-31T23:30"),
                np.timedelta64(30, "m"),
            ),
            pytest.raises(ValueError),
        ),
        (
            np.arange(
                np.datetime64("2010-02-01T00:00"),
                np.datetime64("2010-12-31T23:30"),
                np.timedelta64(61, "m"),
            ),
            pytest.raises(ValueError),
        ),
    ],
)
def test_datetime_check(datetimes, raises):
    """Checks that the datetime checker catches bad datetime ranges."""

    with raises:
        check_datetimes(datetimes)


@pytest.mark.parametrize(
    argnames="inputs, raises, result",
    argvalues=[
        (
            (
                np.ones(365),
                np.arange(
                    np.datetime64("2010-01-01"),
                    np.datetime64("2011-01-01"),
                    np.timedelta64(1, "D"),
                ),
                np.ones(365).astype(bool),
                "total",
            ),
            does_not_raise(),
            [365],
        ),
        (
            (
                np.ones(365),
                np.arange(
                    np.datetime64("2010-01-01"),
                    np.datetime64("2011-01-01"),
                    np.timedelta64(1, "D"),
                ),
                np.ones(365).astype(bool),
                "mean",
            ),
            does_not_raise(),
            [1],
        ),
    ],
)
def test_get_annual(inputs, raises, result):
    """Checks that the get_annual function does the right thing."""

    with raises:
        assert np.allclose(result, get_annual(*inputs))
