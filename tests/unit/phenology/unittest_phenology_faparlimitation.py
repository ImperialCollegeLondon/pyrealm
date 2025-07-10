"""Test the methods used in the FaparLimitation class."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from pyrealm.phenology.fapar_limitation import check_datetimes


@pytest.mark.parametrize(
    argnames="datetimes, raises",
    argvalues=[
        (
            np.arange(
                np.datetime64("2010-01-01T00:00"),
                np.datetime64("2011-01-01T00:00"),
                np.timedelta64(30, "m"),
            ),
            does_not_raise(),
        ),
        (
            np.arange(
                np.datetime64("2010-01-01T00:00"),
                np.datetime64("2010-11-30T23:30"),
                np.timedelta64(30, "m"),
            ),
            pytest.raises(ValueError),
        ),
        (
            np.arange(
                np.datetime64("2010-01-01T00:00"),
                np.datetime64("2010-12-31T23:30"),
                np.timedelta64(61, "m"),
            ),
            pytest.raises(ValueError),
        ),
        (
            np.arange(
                np.datetime64("2010-02-01T00:00"),
                np.datetime64("2010-12-31T23:30"),
                np.timedelta64(30, "m"),
            ),
            pytest.raises(ValueError),
        ),
    ],
)
def test_datetime_check(datetimes, raises):
    """Checks that the datetime checker catches bad datetime ranges."""

    with raises:
        check_datetimes(datetimes)
