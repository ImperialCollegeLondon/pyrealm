"""Tests the implementation of calculations of solar fluxes"""

import numpy as np
import pytest


@pytest.fixture()
def solar_benchmarks(shared_datadir):
    """Test values.

    Loads the input file and solar outputs from the original implementation into numpy
    structured arrays"""

    inputs = np.genfromtxt(
        shared_datadir / "inputs.csv",
        dtype=None,
        delimiter=",",
        names=True,
        encoding="UTF-8",
    )

    expected = np.genfromtxt(
        shared_datadir / "solar_output.csv",
        dtype=None,
        delimiter=",",
        names=True,
        encoding="UTF-8",
    )

    # rename a couple of fields to match new implementation
    exp_fields = list(expected.dtype.names)
    exp_fields[exp_fields.index("my_nu")] = "nu"
    exp_fields[exp_fields.index("my_lambda")] = "lambda_"
    expected.dtype.names = exp_fields

    return inputs, expected


def test_solar_scalars():
    """Tests the results found with a single observation, using inputs from the __main__
    function of the original SPLASH solar.py"""
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar

    cal = Calendar(np.array(["2000-06-20"], dtype="<M8[D]"))

    solar = DailySolarFluxes(
        lat=np.array([37.7]),
        elv=np.array([142]),
        dates=cal,
        sf=np.array([1.0]),
        tc=np.array([23.0]),
    )

    # Output of the __main__ code of original solar.py
    expected = {
        "nu": 166.097934,
        "lambda_": 89.097934,
        "dr": 0.968381,
        "delta": 23.436921,
        "hs": 109.575573,
        "ra_d": 41646763,
        "tau": 0.752844,
        "ppfd_d": 62.042300,
        "rnl": 84.000000,
        "hn": 101.217016,
        "rn_d": 21774953,
        "rnn_d": -3009150,
    }

    for ky, val in expected.items():
        assert np.allclose(getattr(solar, ky), val)


def test_solar_iter(solar_benchmarks):
    """Robust test checking of solar predictions.

    This checks that the outcome of calculating each input row in a time series
    independently gives the same answers as the original implementation, which _has_ to
    iterate over the rows to calculate values.
    """
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar

    inputs, expected = solar_benchmarks

    exp_names = expected.dtype.names

    for day, inp, exp in zip(inputs["dates"], inputs, expected):
        cal = Calendar(np.array([day]).astype("datetime64[D]"))

        solar = DailySolarFluxes(
            lat=inp["lat"], elv=inp["elv"], dates=cal, sf=inp["sf"], tc=inp["tc"]
        )

        for ky in exp_names:
            assert np.allclose(getattr(solar, ky), exp[ky])


def test_solar_array(solar_benchmarks):
    """Array checking of solar predictions.

    This checks that the outcome of calculating all the values in the test inputs
    simultaneously using array inputs gives the same answers as the original
    iterated implementation.
    """
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar

    inputs, expected = solar_benchmarks
    cal = Calendar(inputs["dates"].astype("datetime64[D]"))

    solar = DailySolarFluxes(
        lat=inputs["lat"],
        elv=inputs["elv"],
        dates=cal,
        sf=inputs["sf"],
        tc=inputs["tc"],
    )

    for ky in expected.dtype.names:
        assert np.allclose(getattr(solar, ky), expected[ky])
