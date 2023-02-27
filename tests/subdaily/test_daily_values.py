"""This module tests the interpolator routines for the subdaily model.
"""  # noqa: D205, D415

from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta

import numpy as np
import pytest


@pytest.fixture
def fixture_drv():
    from pyrealm.subdaily import DailyRepresentativeValues

    return DailyRepresentativeValues(
        datetimes=np.arange(
            np.datetime64("2014-06-01 00:00"),
            np.datetime64("2014-06-04 00:00"),
            np.timedelta64(30, "m"),
            dtype="datetime64[s]",
        )
    )


# ----------------------------------------
# Testing DailyRepresentativeValues
# ----------------------------------------


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "datetimes"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "Datetimes are not a 1 dimensional array with dtype datetime64",
            np.arange(0, 144),
            id="Non-datetime64 datetimes",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetimes are not a 1 dimensional array with dtype datetime64",
            np.arange(
                np.datetime64("2014-06-01 00:00"),
                np.datetime64("2014-06-07 00:00"),
                np.timedelta64(30, "m"),
                dtype="datetime64[s]",
            ).reshape((2, 144)),
            id="Non-1D datetimes",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetime sequence not evenly spaced",
            np.datetime64("2014-06-01 12:00")
            + np.cumsum(np.random.randint(25, 35, 144)).astype("timedelta64[m]"),
            id="Uneven sampling",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetime sequence must be increasing",
            np.arange(
                np.datetime64("2014-06-07 00:00"),
                np.datetime64("2014-06-01 00:00"),
                np.timedelta64(-30, "m"),
                dtype="datetime64[s]",
            ),
            id="Negative timedeltas",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetime spacing is not evenly divisible into a day",
            np.arange(
                np.datetime64("2014-06-01 00:00"),
                np.datetime64("2014-06-07 00:00"),
                np.timedelta64(21, "m"),
                dtype="datetime64[s]",
            ),
            id="Spacing not evenly divisible",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetimes include incomplete days",
            np.arange(
                np.datetime64("2014-06-01 12:00"),
                np.datetime64("2014-06-07 00:00"),
                np.timedelta64(30, "m"),
                dtype="datetime64[s]",
            ),
            id="Not complete days by length",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetimes include incomplete days",
            np.arange(
                np.datetime64("2014-06-01 12:00"),
                np.datetime64("2014-06-07 12:00"),
                np.timedelta64(30, "m"),
                dtype="datetime64[s]",
            ),
            id="Not complete days by wrapping",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.arange(
                np.datetime64("2014-06-01 00:00"),
                np.datetime64("2014-06-07 00:00"),
                np.timedelta64(30, "m"),
                dtype="datetime64[s]",
            ),
            id="Correct",
        ),
    ],
)
def test_DRV_init(ctext_mngr, msg, datetimes):
    from pyrealm.subdaily import DailyRepresentativeValues

    with ctext_mngr as cman:
        drep = DailyRepresentativeValues(datetimes=datetimes)

    if msg is not None:
        assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "kwargs"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "window_center and window_width cover more than one day",
            dict(window_center=21, window_width=12),
            id="window > day",
        ),
        pytest.param(
            does_not_raise(),
            None,
            dict(window_center=12, window_width=1),
            id="correct",
        ),
    ],
)
def test_DRV_set_window(fixture_drv, ctext_mngr, msg, kwargs):
    with ctext_mngr as cman:
        fixture_drv.set_window(**kwargs)

    if msg is not None:
        assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "include"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "The include array length is of the wrong length",
            np.ones(76, dtype=np.bool_),
            id="wrong length",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The include argument must be a boolean array",
            np.ones(48),
            id="wrong dtype",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The include argument must be a boolean array",
            "not an array at all",
            id="wrong type",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.ones(48, dtype=np.bool_),
            id="correct",
        ),
    ],
)
def test_DRV_set_include(fixture_drv, ctext_mngr, msg, include):
    with ctext_mngr as cman:
        fixture_drv.set_include(include)

    if msg is not None:
        assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "time"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "The time argument must be a float in (0, 24].",
            "not a time",
            id="not a float",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The time argument must be a float in (0, 24].",
            -1,
            id="time too low",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The time argument must be a float in (0, 24].",
            24,
            id="time too high",
        ),
        pytest.param(
            does_not_raise(),
            None,
            12.0,
            id="correct",
        ),
    ],
)
def test_DRV_set_nearest(fixture_drv, ctext_mngr, msg, time):
    with ctext_mngr as cman:
        fixture_drv.set_nearest(time)

    if msg is not None:
        assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "values"],
    argvalues=[
        (  # Wrong shape
            pytest.raises(ValueError),
            "The first dimension of values is not the same length "
            "as the datetime sequence",
            np.ones((288)),
        ),
    ],
)
def test_DRV_get_rv_errors(fixture_drv, ctext_mngr, msg, values):
    fixture_drv.set_window(window_center=12, window_width=2)

    with ctext_mngr as cman:
        res = fixture_drv.get_representative_values(values)

    assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["values", "expected_means"],
    argvalues=[
        pytest.param(np.ones((3 * 48)), np.array([1, 1, 1]), id="1d_shape_correct"),
        pytest.param(np.ones((3 * 48, 5)), np.ones((3, 5)), id="2d_shape_correct"),
        pytest.param(  # Simple 3D - shape is correct
            np.ones((3 * 48, 5, 5)), np.ones((3, 5, 5)), id="3d_shape_correct"
        ),
        pytest.param(  # 1D - values are correct
            np.arange(144), np.array([24, 72, 120]), id="1d_values_correct"
        ),
        pytest.param(  # 2D - values are correct
            np.broadcast_to(np.arange(144), (5, 144)).T,
            np.tile([24, 72, 120], (5, 1)).T,
            id="2d_values_correct",
        ),
        pytest.param(  # 3D - values are correct
            np.broadcast_to(np.arange(144), (5, 5, 144)).T,
            np.tile([24, 72, 120], (5, 5, 1)).T,
            id="3d_values_correct",
        ),
        pytest.param(  # 3D - values are correct with spatial variation
            np.arange(144 * 25).reshape(144, 5, 5),
            (
                np.tile([600, 1800, 3000], (5, 5, 1)).T
                + np.indices((3, 5, 5))[2]
                + 5 * np.indices((3, 5, 5))[1]
            ),
            id="3d_values_correct_complex",
        ),
    ],
)
class Test_DRV_get_vals:
    """Test DRV get methods.

    This test checks that the correct values are extracted from daily representative
    and that the mean is correctly calculated.
    """

    def test_DRV_get_vals_window(self, fixture_drv, values, expected_means):
        """Test a window"""
        fixture_drv.set_window(window_center=12, window_width=2)
        calculated_means = fixture_drv.get_daily_means(values)

        assert np.allclose(calculated_means, expected_means)

    def test_DRV_get_vals_include(self, fixture_drv, values, expected_means):
        """Test include"""

        # This duplicates the selection of the window test but using direct include
        inc = np.zeros(48, dtype=np.bool_)
        inc[20:29] = True
        fixture_drv.set_include(inc)
        calculated_means = fixture_drv.get_daily_means(values)

        assert np.allclose(calculated_means, expected_means)

    def test_DRV_get_vals_nearest(self, fixture_drv, values, expected_means):
        """Test nearest"""

        # This assumes the data are symmetrical about the middle hour, which is bit of a
        # reach
        fixture_drv.set_nearest(11.8)
        calculated_means = fixture_drv.get_daily_means(values)

        assert np.allclose(calculated_means, expected_means)
