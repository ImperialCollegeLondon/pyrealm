"""This module tests the main functions in the SplashModel:
 - calc_splash_daily, which estimates soil moisture and run off given preceeding soil
   moisture 
 - equilibrate_soil_moisture, which assumes a stationary relationship over an annual
   cycle to estimate a starting soil moisture.
"""

import numpy as np
import pytest
import xarray


@pytest.fixture()
def calc_splash_daily_benchmarks(shared_datadir):
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
        shared_datadir / "run_one_day_output.csv",
        dtype=None,
        delimiter=",",
        names=True,
        encoding="UTF-8",
    )

    return inputs, expected


@pytest.fixture()
def calc_splash_one_d_benchmark(shared_datadir):
    """Test values.

    Loads the input file and resulting soil moisture vector from the original
    implementation."""

    inputs = np.genfromtxt(
        shared_datadir / "example_data.csv",
        dtype=None,
        delimiter=",",
        names=True,
        encoding="UTF-8",
    )

    expected = np.genfromtxt(
        shared_datadir / "example_data_out.csv",
        dtype=None,
        delimiter=",",
        names=True,
        encoding="UTF-8",
    )

    return inputs, expected


@pytest.fixture()
def calc_splash_grid_benchmarks(shared_datadir):
    """Test values.

    Loads the input file and solar outputs from the original implementation into numpy
    structured arrays"""

    inputs = xarray.load_dataset(shared_datadir / "splash_test_grid.nc")

    expected = xarray.load_dataset(shared_datadir / "splash_test_grid_out.nc")

    return inputs, expected


# Testing calc_splash_daily (was run_one_day)


def test_calc_splash_daily():
    from pyrealm.splash.splash import SplashModel
    from pyrealm.splash.utilities import Calendar

    cal = Calendar(np.array("2000-06-20", dtype="<M8[D]"))
    splash = SplashModel(
        lat=np.array(37.7),
        elv=np.array(142),
        sf=np.array(1.0),
        tc=np.array(23.0),
        pn=np.array(5),
        dates=cal,
    )
    sm, ro = splash.calc_splash_daily(wn=np.array(75))

    # Output of __main__ in original splash.py
    evap_expected = {
        "cond": 0.885192,
        "eet_d": 6.405468,
        "pet_d": 8.070889,
        "aet_d": 5.748034,
    }

    for ky, val in evap_expected.items():
        assert np.allclose(getattr(splash.evap, ky), val)

    assert np.allclose(sm, 75.137158)
    assert np.allclose(ro, 0.0000000)


def test_calc_splash_daily_iter(calc_splash_daily_benchmarks):
    from pyrealm.splash.splash import SplashModel
    from pyrealm.splash.utilities import Calendar

    inputs, expected = calc_splash_daily_benchmarks
    cal = Calendar(inputs["dates"].astype("datetime64[D]"))

    evap_expected = [
        ("cond", "cond"),
        ("eet_d", "eet"),
        ("pet_d", "pet"),
        ("aet_d", "aet"),
    ]

    for day, inp, exp in zip(cal, inputs, expected):
        # initialise splash and calculate the evaporative flux and soil moisture
        splash = SplashModel(
            lat=inp["lat"],
            elv=inp["elv"],
            dates=day,
            sf=inp["sf"],
            tc=inp["tc"],
            pn=inp["pn"],
        )
        sm, ro = splash.calc_splash_daily(wn=inp["wn"])

        for ky1, ky2 in evap_expected:
            assert np.allclose(getattr(splash.evap, ky1), exp[ky2])

        assert np.allclose(sm, exp["wn"])
        assert np.allclose(ro, exp["ro"])


def test_calc_splash_daily_array(calc_splash_daily_benchmarks):
    from pyrealm.splash.splash import SplashModel
    from pyrealm.splash.utilities import Calendar

    inputs, expected = calc_splash_daily_benchmarks
    cal = Calendar(inputs["dates"].astype("datetime64[D]"))

    evap_expected = [
        ("cond", "cond"),
        ("eet_d", "eet"),
        ("pet_d", "pet"),
        ("aet_d", "aet"),
    ]

    splash = SplashModel(
        lat=inputs["lat"],
        elv=inputs["elv"],
        dates=cal,
        sf=inputs["sf"],
        tc=inputs["tc"],
        pn=inputs["pn"],
    )
    sm, ro = splash.calc_splash_daily(wn=inputs["wn"])

    for ky1, ky2 in evap_expected:
        assert np.allclose(getattr(splash.evap, ky1), expected[ky2])

    assert np.allclose(sm, expected["wn"])
    assert np.allclose(ro, expected["ro"])


# Testing the spin-up process


def test_run_spin_up_oned(calc_splash_one_d_benchmark):
    from pyrealm.splash.splash import SplashModel
    from pyrealm.splash.utilities import Calendar

    inputs, expected = calc_splash_one_d_benchmark

    dates = Calendar(
        np.arange(np.datetime64("2000-01-01"), np.datetime64("2001-01-01"))
    )
    splash = SplashModel(
        lat=np.array([37.7]),
        elv=np.array([142.0]),
        dates=dates,
        sf=inputs["sf"],
        tc=inputs["tair"],
        pn=inputs["pn"],
    )

    sm, ro = splash.equilibrate_soil_moisture()
