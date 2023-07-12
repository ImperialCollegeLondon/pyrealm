"""This module tests the main functions in the SplashModel:
 - calc_splash_daily, which estimates soil moisture and run off given preceeding soil
   moisture 
 - equilibrate_soil_moisture, which assumes a stationary relationship over an annual
   cycle to estimate a starting soil moisture.
"""

import numpy as np
import pytest
import xarray
from splash_fixtures import daily_flux_benchmarks, grid_benchmarks


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


# Testing calc_splash_daily (was run_one_day)


def test_estimate_daily_water_balance_scalar():
    """Tests a single day calculation aganist the expectations from the __main__ example
    provided in SPLASH v1.0 splash.py"""
    from pyrealm.splash.splash import SplashModel
    from pyrealm.splash.utilities import Calendar

    cal = Calendar(np.array(["2000-06-20"], dtype="<M8[D]"))
    splash = SplashModel(
        lat=np.array([37.7]),
        elv=np.array([142]),
        sf=np.array([1.0]),
        tc=np.array([23.0]),
        pn=np.array([5]),
        dates=cal,
    )
    aet, sm, ro = splash.estimate_daily_water_balance(
        previous_wn=np.array(75), day_idx=0
    )

    # Expected values are the output of __main__ in original splash.py
    evap_expected = {
        "cond": 0.885192,
        "eet_d": 6.405468,
        "pet_d": 8.070889,
    }

    for ky, val in evap_expected.items():
        assert np.allclose(getattr(splash.evap, ky), val)

    assert np.allclose(aet, 5.748034)
    assert np.allclose(sm, 75.137158)
    assert np.allclose(ro, 0.0000000)


def test_estimate_daily_water_balance_iter(daily_flux_benchmarks):
    """This test iterates over the individual daily benchmark rows, calculating each
    prediction as a single independent day."""
    from pyrealm.splash.splash import SplashModel
    from pyrealm.splash.utilities import Calendar

    inputs, expected = daily_flux_benchmarks
    days = inputs["dates"].astype("datetime64[D]")

    for day, inp, exp in zip(days, inputs, expected):
        # initialise splash and calculate the evaporative flux and soil moisture
        splash = SplashModel(
            lat=np.array([inp["lat"]]),
            elv=np.array([inp["elv"]]),
            sf=np.array([inp["sf"]]),
            tc=np.array([inp["tc"]]),
            pn=np.array([inp["pn"]]),
            dates=Calendar(np.array([day])),
        )
        aet, sm, ro = splash.estimate_daily_water_balance(
            previous_wn=np.array([inp["wn"]]), day_idx=0
        )
        assert np.allclose(aet, exp["aet_d"])
        # assert np.allclose(sm, exp["wn"])
        # assert np.allclose(ro, exp["ro"])


def test_estimate_daily_water_balance_array(daily_flux_benchmarks):
    """This test runs the individual daily benchmark data as an array."""
    from pyrealm.splash.splash import SplashModel
    from pyrealm.splash.utilities import Calendar

    inputs, expected = daily_flux_benchmarks

    splash = SplashModel(
        lat=inputs["lat"],
        elv=inputs["elv"],
        sf=inputs["sf"],
        tc=inputs["tc"],
        pn=inputs["pn"],
        dates=Calendar(inputs["dates"]),
    )

    aet, sm, ro = splash.estimate_daily_water_balance(
        previous_wn=inputs["wn"], day_idx=None
    )

    assert np.allclose(aet, expected["aet_d"])


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


def test_splashmodel_est_daily_soil_moisture(grid_benchmarks):
    """Array checking of evaporative predictions using iteration over days.

    This checks that the outcome of evaporative calculations from running the full
    SPLASH model on a gridded dataset are consistent.
    """
    from pyrealm.constants import PModelConst
    from pyrealm.pmodel.functions import calc_patm
    from pyrealm.splash.splash import SplashModel, elv2pres
    from pyrealm.splash.utilities import Calendar

    inputs, expected = grid_benchmarks

    cal = Calendar(inputs.time.values.astype("datetime64[D]"))

    # Duplicate lat and elev to same shape as sf and tc (TODO - avoid this!)
    sf_shape = inputs["sf"].shape
    elev = np.repeat(inputs["elev"].data[np.newaxis, :, :], sf_shape[0], axis=0)
    lat = np.repeat(inputs["lat"].data[:, np.newaxis], sf_shape[2], axis=1)
    lat = np.repeat(lat[np.newaxis, :, :], sf_shape[0], axis=0)

    splash = SplashModel(
        lat=lat,
        elv=elev,
        sf=inputs["sf"].data,
        pn=inputs["pre"].data,
        tc=inputs["tmp"].data,
        dates=cal,
    )

    # assert the same starting point as the original spun up state
    curr_wn = expected["wn_spun_up"].data

    # For each day, calculate the estimated soil moisture and run off
    for day_idx, _ in enumerate(cal):
        curr_wn, ro = splash.estimate_daily_soil_moisture(
            previous_wn=curr_wn, day_index=day_idx
        )
        assert np.allclose(curr_wn, expected["wn"][day_idx])
        assert np.allclose(ro, expected["ro"][day_idx])
