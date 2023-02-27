# flake8: noqa D103 - docstrings on unit tests

"""This module tests the interpolator routines for the subdaily model.
"""  # noqa: D205, D415

from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta

import numpy as np
import pytest

from pyrealm.subdaily.interpolators import TemporalInterpolator

# ----------------------------------------
# Testing TemporalInterpolator
# ----------------------------------------


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "input", "interp"],
    argvalues=[
        (  # Two non-datetime64 inputs
            pytest.raises(TypeError),
            "Interpolation times must be np.datetime64 arrays",
            np.arange(0, 144),
            np.arange(0, 144),
        ),
        (  # One non-datetime64 inputs
            pytest.raises(TypeError),
            "Interpolation times must be np.datetime64 arrays",
            np.arange(0, 144),
            np.arange(
                datetime(2014, 6, 1, 0, 0),
                datetime(2014, 6, 7, 0, 0),
                timedelta(minutes=30),
                dtype="datetime64[m]",
            ),
        ),
        (  # Datetime64 inputs with different precision
            pytest.raises(TypeError),
            "Inputs must use the same np.datetime64 precision subtype",
            np.arange(
                datetime(2014, 6, 1, 0, 0),
                datetime(2014, 6, 7, 0, 0),
                timedelta(days=1),
                dtype="datetime64[m]",
            ),
            np.arange(
                datetime(2014, 6, 1, 0, 0),
                datetime(2014, 6, 7, 0, 0),
                timedelta(minutes=30),
                dtype="datetime64[s]",
            ),
        ),
    ],
)
def test_temporal_interpolator_init_errors(ctext_mngr, msg, input, interp):
    with ctext_mngr as cman:
        tint = TemporalInterpolator(
            input_datetimes=input, interpolation_datetimes=interp
        )

    assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "values"],
    argvalues=[
        (  # 1D - Incorrect length on axis 0
            pytest.raises(ValueError),
            "The first axis of values does not match the length of input_datetimes",
            np.arange(0, 144),
        ),
    ],
)
def test_temporal_interpolator_call_errors(ctext_mngr, msg, values):
    tint = TemporalInterpolator(
        input_datetimes=np.arange(
            datetime(2014, 6, 1, 0, 0),
            datetime(2014, 6, 7, 0, 0),
            timedelta(days=1),
            dtype="datetime64[m]",
        ),
        interpolation_datetimes=np.arange(
            datetime(2014, 6, 1, 0, 0),
            datetime(2014, 6, 7, 0, 0),
            timedelta(minutes=30),
            dtype="datetime64[m]",
        ),
    )

    with ctext_mngr as cman:
        tint(values)

    assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["method", "values", "expected"],
    argvalues=[
        (  # Simple 1D linear - shape
            "linear",
            np.ones((4)),
            np.ones((145)),
        ),
        (  # 1D linear - values
            "linear",
            np.arange(1, 5),
            np.linspace(1, 4, 145),
        ),
        (  # 2D linear - values
            "linear",
            np.broadcast_to(np.arange(1, 5), (5, 4)).T,
            np.broadcast_to(np.linspace(1, 4, 145), (5, 145)).T,
        ),
        (  # 3D linear - values
            "linear",
            np.broadcast_to(np.arange(1, 5), (5, 5, 4)).T,
            np.broadcast_to(np.linspace(1, 4, 145), (5, 5, 145)).T,
        ),
        (  # 3D - values are correct with spatial variation
            "linear",
            np.arange(1, 5 * 5 * 4 + 1).reshape(5, 5, 4).T,
            (
                np.broadcast_to(np.linspace(1, 4, 145), (5, 5, 145)).T
                + 20 * np.indices((145, 5, 5))[2]
                + 4 * np.indices((145, 5, 5))[1]
            ),
        ),
        # (  # Simple 1D daily_constant - values
        #     'daily_constant',
        #     np.arange(1, 5),
        #     np.repeat(np.arange(1, 5), 48)
        # ),
    ],
)
def test_temporal_interpolator_call(method, values, expected):
    tint = TemporalInterpolator(
        input_datetimes=np.arange(
            datetime(2014, 6, 1, 0, 0),
            datetime(2014, 6, 4, 0, 1),
            timedelta(days=1),
            dtype="datetime64[m]",
        ),
        interpolation_datetimes=np.arange(
            datetime(2014, 6, 1, 0, 0),
            datetime(2014, 6, 4, 0, 1),
            timedelta(minutes=30),
            dtype="datetime64[m]",
        ),
        method=method,
    )

    calculated = tint(values)

    assert np.allclose(calculated, expected)


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
    from pyrealm.subdaily.interpolators import DailyRepresentativeValues

    with ctext_mngr as cman:
        drep = DailyRepresentativeValues(datetimes=datetimes)

    if msg is not None:
        assert str(cman.value) == msg


@pytest.fixture
def fixture_drv():
    from pyrealm.subdaily.interpolators import DailyRepresentativeValues

    return DailyRepresentativeValues(
        datetimes=np.arange(
            np.datetime64("2014-06-01 00:00"),
            np.datetime64("2014-06-04 00:00"),
            np.timedelta64(30, "m"),
            dtype="datetime64[s]",
        )
    )


@pytest.fixture
def fixture_drv_with_window(fixture_drv):
    fixture_drv.set_window(window_center=12, window_width=2)
    return fixture_drv


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
        # (  # Include is the wrong shape
        #     pytest.raises(ValueError),
        #     "Datetimes and include do not have the same shape",
        #     dict(include=np.ones((288), dtype="bool")),
        # ),
        # (  # Include is the wrong type
        #     pytest.raises(ValueError),
        #     "The include argument must be a boolean array",
        #     np.arange(
        #         datetime(2014, 6, 1, 0, 0),
        #         datetime(2014, 6, 4, 0, 0),
        #         timedelta(minutes=30),
        #         dtype="datetime64[m]",
        #     ),
        #     dict(include=np.ones((144))),
        # ),
    ],
)
def test_DRV_set_window(fixture_drv, ctext_mngr, msg, kwargs):
    with ctext_mngr as cman:
        fixture_drv.set_window(**kwargs)

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
def test_daily_representative_values_call_errors(
    fixture_drv_with_window, ctext_mngr, msg, values
):
    with ctext_mngr as cman:
        res = fixture_drv_with_window.get_representative_values(values)

    assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["values", "expected"],
    argvalues=[
        pytest.param(  # Simple 1D - shape is correct
            np.ones((3 * 48)),
            np.array([1, 1, 1]),
        ),
        (  # Simple 2D - shape is correct
            np.ones((3 * 48, 5)),
            np.ones((3, 5)),
        ),
        (  # Simple 3D - shape is correct
            np.ones((3 * 48, 5, 5)),
            np.ones((3, 5, 5)),
        ),
        (  # 1D - values are correct
            np.arange(144),
            np.array([24, 72, 120]),
        ),
        (  # 2D - values are correct
            np.broadcast_to(np.arange(144), (5, 144)).T,
            np.tile([24, 72, 120], (5, 1)).T,
        ),
        (  # 3D - values are correct
            np.broadcast_to(np.arange(144), (5, 5, 144)).T,
            np.tile([24, 72, 120], (5, 5, 1)).T,
        ),
        (  # 3D - values are correct with spatial variation
            np.arange(144 * 25).reshape(144, 5, 5),
            (
                np.tile([600, 1800, 3000], (5, 5, 1)).T
                + np.indices((3, 5, 5))[2]
                + 5 * np.indices((3, 5, 5))[1]
            ),
        ),
    ],
)
def test_DRV_get_rv_and_daily_means(fixture_drv_with_window, values, expected):
    calculated = fixture_drv_with_window.get_daily_means(values)

    assert np.allclose(calculated, expected)
