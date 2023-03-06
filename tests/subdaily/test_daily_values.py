"""This module tests the interpolator routines for the subdaily model.
"""  # noqa: D205, D415

from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta

import numpy as np
import pytest


@pytest.fixture
def fixture_FSS():
    from pyrealm.subdaily import FastSlowScaler

    return FastSlowScaler(
        datetimes=np.arange(
            np.datetime64("2014-06-01 00:00"),
            np.datetime64("2014-06-04 00:00"),
            np.timedelta64(30, "m"),
            dtype="datetime64[s]",
        )
    )


# ----------------------------------------
# Testing FastSlowScaler
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
def test_FSS_init(ctext_mngr, msg, datetimes):
    from pyrealm.subdaily import FastSlowScaler

    with ctext_mngr as cman:
        drep = FastSlowScaler(datetimes=datetimes)

    if msg is not None:
        assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "kwargs", "samp_mean", "samp_max"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "window_center and window_width must be np.timedelta64 values",
            dict(window_center=21, half_width=12),
            None,
            None,
            id="not np.timedeltas",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "window_center and window_width cover more than one day",
            dict(
                window_center=np.timedelta64(21, "h"),
                half_width=np.timedelta64(6, "h"),
            ),
            None,
            None,
            id="window > day",
        ),
        pytest.param(
            does_not_raise(),
            None,
            dict(
                window_center=np.timedelta64(12, "h"),
                half_width=np.timedelta64(1, "h"),
            ),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([1, 25, 49], dtype="timedelta64[h]"),
            id="correct",
        ),
    ],
)
def test_FSS_set_window(fixture_FSS, ctext_mngr, msg, kwargs, samp_mean, samp_max):
    with ctext_mngr as cman:
        fixture_FSS.set_window(**kwargs)

    if msg is not None:
        assert str(cman.value) == msg
    else:
        # Check that _set_times has run correctly. Can't use allclose directly on
        # datetimes and since these are integers under the hood, don't need float
        # testing
        assert np.all(fixture_FSS.sample_datetimes_mean == samp_mean)
        assert np.all(fixture_FSS.sample_datetimes_max == samp_max)


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "include", "samp_mean", "samp_max"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "The include array length is of the wrong length",
            np.ones(76, dtype=np.bool_),
            None,
            None,
            id="wrong length",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The include argument must be a boolean array",
            np.ones(48),
            None,
            None,
            id="wrong dtype",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The include argument must be a boolean array",
            "not an array at all",
            None,
            None,
            id="wrong type",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.repeat([False, True, False], (22, 5, 21)),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([1, 25, 49], dtype="timedelta64[h]"),
            id="correct - noon window",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.ones(48, dtype=np.bool_),
            np.datetime64("2014-06-01 11:45:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            np.datetime64("2014-06-01 11:30:00")
            + np.array([12, 36, 60], dtype="timedelta64[h]"),
            id="correct - whole day",
        ),
    ],
)
def test_FSS_set_include(fixture_FSS, ctext_mngr, msg, include, samp_mean, samp_max):
    with ctext_mngr as cman:
        fixture_FSS.set_include(include)

    if msg is not None:
        assert str(cman.value) == msg

    else:
        # Check that _set_times has run correctly. Can't use allclose directly on
        # datetimes and since these are integers under the hood, don't need float
        # testing
        assert np.all(fixture_FSS.sample_datetimes_mean == samp_mean)
        assert np.all(fixture_FSS.sample_datetimes_max == samp_max)


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "time", "samp_mean", "samp_max"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "The time argument must be a timedelta64 value.",
            "not a time",
            None,
            None,
            id="string input",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The time argument must be a timedelta64 value.",
            12,
            None,
            None,
            id="float input",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The time argument is not >= 0 and < 24 hours.",
            np.timedelta64(-1, "h"),
            None,
            None,
            id="time too low",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The time argument is not >= 0 and < 24 hours.",
            np.timedelta64(24, "h"),
            None,
            None,
            id="time too high",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.timedelta64(12, "h"),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            id="correct",
        ),
    ],
)
def test_FSS_set_nearest(fixture_FSS, ctext_mngr, msg, time, samp_mean, samp_max):
    with ctext_mngr as cman:
        fixture_FSS.set_nearest(time)

    if msg is not None:
        assert str(cman.value) == msg

    else:
        # Check that _set_times has run correctly. Can't use allclose directly on
        # datetimes and since these are integers under the hood, don't need float
        # testing
        assert np.all(fixture_FSS.sample_datetimes_mean == samp_mean)
        assert np.all(fixture_FSS.sample_datetimes_max == samp_max)


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
def test_FSS_get_rv_errors(fixture_FSS, ctext_mngr, msg, values):
    fixture_FSS.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(2, "h"),
    )

    with ctext_mngr as cman:
        res = fixture_FSS.get_representative_values(values)

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
class Test_FSS_get_vals:
    """Test FSS get methods.

    This test checks that the correct values are extracted from daily representative
    and that the mean is correctly calculated.
    """

    def test_FSS_get_vals_window(self, fixture_FSS, values, expected_means):
        """Test a window"""
        fixture_FSS.set_window(
            window_center=np.timedelta64(12, "h"),
            half_width=np.timedelta64(2, "h"),
        )
        calculated_means = fixture_FSS.get_daily_means(values)

        assert np.allclose(calculated_means, expected_means)

    def test_FSS_get_vals_include(self, fixture_FSS, values, expected_means):
        """Test include"""

        # This duplicates the selection of the window test but using direct include
        inc = np.zeros(48, dtype=np.bool_)
        inc[20:29] = True
        fixture_FSS.set_include(inc)
        calculated_means = fixture_FSS.get_daily_means(values)

        assert np.allclose(calculated_means, expected_means)

    def test_FSS_get_vals_nearest(self, fixture_FSS, values, expected_means):
        """Test nearest"""

        # This assumes the data are symmetrical about the middle hour, which is bit of a
        # reach
        fixture_FSS.set_nearest(11.8)
        calculated_means = fixture_FSS.get_daily_means(values)

        assert np.allclose(calculated_means, expected_means)


@pytest.mark.parametrize(
    argnames=["method_name", "kwargs", "update_point"],
    argvalues=[
        pytest.param(
            "set_window",
            dict(
                window_center=np.timedelta64(12, "h"),
                half_width=np.timedelta64(1, "h"),
            ),
            "max",
            id="window_max",
        ),
        pytest.param(
            "set_window",
            dict(
                window_center=np.timedelta64(13, "h"),
                half_width=np.timedelta64(1, "h"),
            ),
            "mean",
            id="window_mean",
        ),
        pytest.param(
            "set_include",
            dict(
                include=np.repeat([False, True, False], (22, 5, 21)),
            ),
            "max",
            id="include_max",
        ),
        pytest.param(
            "set_include",
            dict(
                include=np.repeat([False, True, False], (24, 5, 19)),
            ),
            "mean",
            id="include_mean",
        ),
        pytest.param(
            "set_nearest",
            dict(
                time=np.timedelta64(13, "h"),
            ),
            "max",
            id="nearest_max",
        ),
        pytest.param(
            "set_nearest",
            dict(
                time=np.timedelta64(13, "h"),
            ),
            "mean",
            id="nearest_mean",
        ),
    ],
)
@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "input_values", "exp_values"],
    argvalues=[
        pytest.param(
            does_not_raise(),
            None,
            np.array([1, 2, 3]),
            np.repeat([np.nan, 1, 2, 3], (26, 48, 48, 22)),
            id="1D test",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.array([[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]),
            np.repeat(
                a=[
                    [[np.nan, np.nan], [np.nan, np.nan]],
                    [[1, 4], [7, 10]],
                    [[2, 5], [8, 11]],
                    [[3, 6], [9, 12]],
                ],
                repeats=[26, 48, 48, 22],
                axis=0,
            ),
            id="2D test",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.array([[1, 4], [2, 5], [3, 6]]),
            np.repeat(
                a=[[np.nan, np.nan], [1, 4], [2, 5], [3, 6]],
                repeats=[26, 48, 48, 22],
                axis=0,
            ),
            id="3D test",
        ),
    ],
)
def test_FSS_resample_subdaily(
    fixture_FSS,
    method_name,
    kwargs,
    update_point,
    ctext_mngr,
    msg,
    input_values,
    exp_values,
):
    # Set the included observations - the different parameterisations here and for
    # the update point should all select the same update point.
    func = getattr(fixture_FSS, method_name)
    func(**kwargs)

    with ctext_mngr as cman:
        res = fixture_FSS.fill_daily_to_subdaily(
            input_values, update_point=update_point
        )

    if cman is not None:
        assert str(cman.value) == msg
    else:
        assert np.allclose(res, exp_values, equal_nan=True)
