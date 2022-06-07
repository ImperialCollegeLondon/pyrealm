# flake8: noqa D103 - docstrings on unit tests

from datetime import datetime, timedelta
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pytest

from pyrealm.utilities import DailyRepresentativeValues


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "datetimes", "kwargs"],
    argvalues=[
        (  # Non-datetime64 datetimes
            pytest.raises(ValueError),
            "Datetimes are not a 1 dimensional array with dtype datetime64",
            np.arange(0, 144),
            dict(window_center=12, window_width=2),
        ),
        (  # 2 dimensionsal datetimes
            pytest.raises(ValueError),
            "Datetimes are not a 1 dimensional array with dtype datetime64",
            np.arange(
                datetime(2014, 6, 1, 0, 0),
                datetime(2014, 6, 7, 0, 0),
                timedelta(minutes=30),
                dtype="datetime64[m]",
            ).reshape((2, 144)),
            dict(window_center=12, window_width=2),
        ),
        (  # Uneven temporal sampling
            pytest.raises(ValueError),
            "Datetime sequence must be evenly spaced",
            np.array(
                [
                    datetime(2014, 6, 1, 0, 0) + timedelta(minutes=int(d))
                    for d in np.cumsum(np.random.randint(25, 35, 144))
                ],
                dtype="datetime64[m]",
            ),
            dict(window_center=12, window_width=2),
        ),
        (  # Negative timedeltas
            pytest.raises(ValueError),
            "Datetime sequence must be increasing",
            np.arange(
                datetime(2014, 6, 7, 0, 0),
                datetime(2014, 6, 1, 0, 0),
                timedelta(minutes=-30),
                dtype="datetime64[m]",
            ),
            dict(window_center=12, window_width=2),
        ),
        (  # Negative timedeltas
            pytest.raises(ValueError),
            "Datetime sequence does not cover a whole number of days",
            np.arange(
                datetime(2014, 6, 1, 10, 1),
                datetime(2014, 6, 7, 22, 46),
                timedelta(minutes=30),
                dtype="datetime64[m]",
            ),
            dict(window_center=12, window_width=2),
        ),
        (  # Window greater than one day
            pytest.raises(NotImplementedError),
            "window_center and window_width cover more than one day",
            np.arange(
                datetime(2014, 6, 1, 0, 0),
                datetime(2014, 6, 4, 0, 0),
                timedelta(minutes=30),
                dtype="datetime64[m]",
            ),
            dict(window_center=21, window_width=12),
        ),
        (  # Include is the wrong shape
            pytest.raises(ValueError),
            "Datetimes and include do not have the same shape",
            np.arange(
                datetime(2014, 6, 1, 0, 0),
                datetime(2014, 6, 4, 0, 0),
                timedelta(minutes=30),
                dtype="datetime64[m]",
            ),
            dict(include=np.ones((288), dtype="bool")),
        ),
        (  # Include is the wrong type
            pytest.raises(ValueError),
            "The include argument must be a boolean array",
            np.arange(
                datetime(2014, 6, 1, 0, 0),
                datetime(2014, 6, 4, 0, 0),
                timedelta(minutes=30),
                dtype="datetime64[m]",
            ),
            dict(include=np.ones((144))),
        ),
    ],
)
def test_daily_representative_values_init_errors(ctext_mngr, msg, datetimes, kwargs):

    with ctext_mngr as cman:

        drep = DailyRepresentativeValues(datetimes=datetimes, **kwargs)

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
def test_daily_representative_values_call_errors(ctext_mngr, msg, values):

    drep = DailyRepresentativeValues(
        datetimes=np.arange(
            datetime(2014, 6, 1, 0, 0),
            datetime(2014, 6, 4, 0, 0),
            timedelta(minutes=30),
            dtype="datetime64[m]",
        ),
        window_center=12,
        window_width=2,
    )

    with ctext_mngr as cman:

        res = drep(values)

    assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["kwargs", "values", "expected"],
    argvalues=[
        (  # Simple 1D - shape is correct
            dict(window_center=12, window_width=2),
            np.ones((3 * 48)),
            np.array([1, 1, 1]),
        ),
        (  # Simple 2D - shape is correct
            dict(window_center=12, window_width=2),
            np.ones((3 * 48, 5)),
            np.ones((3, 5)),
        ),
        (  # Simple 3D - shape is correct
            dict(window_center=12, window_width=2),
            np.ones((3 * 48, 5, 5)),
            np.ones((3, 5, 5)),
        ),
        (  # 1D - values are correct
            dict(window_center=12, window_width=2),
            np.arange(144),
            np.array([24, 72, 120]),
        ),
        (  # 2D - values are correct
            dict(window_center=12, window_width=2),
            np.broadcast_to(np.arange(144), (5, 144)).T,
            np.tile([24, 72, 120], (5, 1)).T,
        ),
        (  # 3D - values are correct
            dict(window_center=12, window_width=2),
            np.broadcast_to(np.arange(144), (5, 5, 144)).T,
            np.tile([24, 72, 120], (5, 5, 1)).T,
        ),
        (  # 3D - values are correct with spatial variation
            dict(window_center=12, window_width=2),
            np.arange(144 * 25).reshape(144, 5, 5),
            (np.tile([600, 1800, 3000], (5, 5, 1)).T
             + np.indices((3, 5, 5))[2]
             + 5 * np.indices((3, 5, 5))[1]),
        ),
        (  # 1D - include - ragged array using exponential series to get more
            # sensitive test than just one or unity increments and to test the
            # division by count aligns correctly
            dict(
                include=np.concatenate(
                    [
                        [True] * 5,
                        [False] * 43,
                        [True] * 9,
                        [False] * 39,
                        [True] * 13,
                        [False] * 35,
                    ],
                    dtype=bool,
                )
            ),
            np.power(np.arange(1, 145), 0.2),
            np.array(
                [
                    np.mean(np.power(np.arange(1, 1 + 5), 0.2)),
                    np.mean(np.power(np.arange(49, 49 + 9), 0.2)),
                    np.mean(np.power(np.arange(97, 97 + 13), 0.2)),
                ]
            ),
        ),
    ],
)
def test_daily_representative_values_call(kwargs, values, expected):

    drep = DailyRepresentativeValues(
        datetimes=np.arange(
            datetime(2014, 6, 1, 0, 0),
            datetime(2014, 6, 4, 0, 0),
            timedelta(minutes=30),
            dtype="datetime64[m]",
        ),
        **kwargs
    )

    calculated = drep(values)

    assert np.allclose(calculated, expected)
