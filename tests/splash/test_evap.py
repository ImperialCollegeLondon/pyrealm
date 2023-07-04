"""Testing the evaporative flux calculations against benchmark data from the original
SPLASH calculations"""

import numpy as np
import pytest


@pytest.fixture()
def evap_benchmarks(shared_datadir):
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
        shared_datadir / "evap_output.csv",
        dtype=None,
        delimiter=",",
        names=True,
        encoding="UTF-8",
    )

    return inputs, expected


def test_evap_scalar():
    """Test using array inputs with a single scalar value. The expected results are as
    the original output from the SPLASH evap.py __main__ function"""
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar

    cal = Calendar(np.array("2000-06-20", dtype="<M8[D]"))
    solar = DailySolarFluxes(
        lat=np.array([37.7]),
        elv=np.array([142]),
        dates=cal,
        sf=np.array([1.0]),
        tc=np.array([23.0]),
    )

    evap = DailyEvapFluxes(solar, pa=np.array([99630.833]), tc=np.array([23.0]))
    aet, hi = evap.estimate_aet(sw=np.array([0.9]), return_hi=True)

    # Output of __main__ code in original evap.py
    expected = {
        "sat": 169.89609255250576,
        "lv": 2446686.637327215,
        "pw": 997.5836204018437,
        "psy": 66.72971923515009,
        "econ": 2.941667713784511e-10,
        "cond": 0.8851919575664212,
        "eet_d": 6.405467536773751,
        "pet_d": 8.070889096334925,
        "rx": 0.0013343404749726541,
    }

    for ky, val in expected.items():
        assert np.allclose(getattr(evap, ky), val)

    assert np.allclose(aet, 7.972787573253663)
    assert np.allclose(hi, 20.95931970358043)


def test_evap_iter(evap_benchmarks):
    """Robust test checking of evap predictions.

    This checks that the outcome of calculating each input row in a time series
    independently gives the same answers as the original implementation, which _has_ to
    iterate over the rows to calculate values.
    """
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar

    inputs, expected = evap_benchmarks

    exp_names = expected.dtype.names

    for day, inp, exp in zip(inputs["dates"], inputs, expected):
        # Convert the input row into a dictionary of 1D arrays
        inp = {nm: np.array([inp[nm]]) for nm in inputs.dtype.names}

        cal = Calendar(np.array([day]).astype("datetime64[D]"))
        solar = DailySolarFluxes(
            lat=inp["lat"], elv=inp["elv"], dates=cal, sf=inp["sf"], tc=inp["tc"]
        )

        evap = DailyEvapFluxes(solar, pa=inp["pa"], tc=inp["tc"])
        aet, hi = evap.estimate_aet(sw=inp["sw"], return_hi=True)

        expected_attr = set(exp_names) - {"aet_d", "hi"}
        for ky in expected_attr:
            assert np.allclose(getattr(evap, ky), exp[ky])

        assert np.allclose(aet, exp["aet_d"])
        assert np.allclose(hi, exp["hi"])


def test_evap_array(evap_benchmarks):
    """Array checking of evaporative predictions.

    This checks that the outcome of calculating all the values in the test inputs
    _simultaneously_ using array inputs gives the same answers as the original
    iterated implementation.
    """
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar

    inputs, expected = evap_benchmarks
    cal = Calendar(inputs["dates"].astype("datetime64[D]"))

    exp_names = expected.dtype.names

    solar = DailySolarFluxes(
        lat=inputs["lat"],
        elv=inputs["elv"],
        dates=cal,
        sf=inputs["sf"],
        tc=inputs["tc"],
    )

    evap = DailyEvapFluxes(solar, pa=inputs["pa"], tc=inputs["tc"])
    aet, hi = evap.estimate_aet(sw=inputs["sw"], return_hi=True)

    expected_attr = set(exp_names) - {"aet_d", "hi"}
    for ky in expected_attr:
        assert np.allclose(getattr(evap, ky), expected[ky])

    assert np.allclose(aet, expected["aet_d"])
    assert np.allclose(hi, expected["hi"])


# TODO - test the day index approach.
