# flake8: noqa D103 - docstrings on unit tests

"""This module tests the interpolator routines for the subdaily model.
"""  # noqa: D205, D415

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
