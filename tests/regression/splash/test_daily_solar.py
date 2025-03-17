"""Tests the implementation of calculations of solar fluxes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def expected_attr():
    """Returns the names of the attributes of DailySolarFluxes to test."""

    return (
        "nu",
        "lambda_",
        "distance_factor",
        "declination",
        "sunset_hour_angle",
        "daily_solar_radiation",
        "transmissivity",
        "ppfd_d",
        "net_longwave_radiation",
        "crossover_hour_angle",
        "daytime_net_radiation",
        "nighttime_net_radiation",
    )


def test_solar_scalars():
    """Tests the results found with a single observation.

    Uses using inputs from the __main__ function of the original SPLASH solar.py
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.solar import DailySolarFluxes

    cal = Calendar(np.array(["2000-06-20"], dtype="<M8[D]"))

    solar = DailySolarFluxes(
        latitude=np.array([37.7]),
        elevation=np.array([142]),
        dates=cal,
        sunshine_fraction=np.array([1.0]),
        temperature=np.array([23.0]),
    )

    # Output of the __main__ code of original solar.py
    expected = {
        "nu": 166.097934,
        "lambda_": 89.097934,
        "distance_factor": 0.968381,
        "declination": 23.436921,
        "sunset_hour_angle": 109.575573,
        "daily_solar_radiation": 41646763,
        "transmissivity": 0.752844,
        "ppfd_d": 62.042300,
        "net_longwave_radiation": 84.000000,
        "crossover_hour_angle": 101.217016,
        "daytime_net_radiation": 21774953,
        "nighttime_net_radiation": -3009150,
    }

    for ky, val in expected.items():
        assert_allclose(getattr(solar, ky), val, rtol=1e-6)


def test_solar_iter(daily_flux_benchmarks, expected_attr):
    """Robust test checking of solar predictions.

    This checks that the outcome of calculating each input row in a time series
    independently gives the same answers as the original implementation, which _has_ to
    iterate over the rows to calculate values.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = daily_flux_benchmarks

    for day, (_, inp), (_, exp) in zip(
        inputs["dates"], inputs.iterrows(), expected.iterrows()
    ):
        cal = Calendar(np.array([day]).astype("datetime64[D]"))

        solar = DailySolarFluxes(
            dates=cal,
            latitude=np.array([inp["lat"]]),
            elevation=np.array([inp["elv"]]),
            sunshine_fraction=np.array([inp["sf"]]),
            temperature=np.array([inp["tc"]]),
        )

        for ky in expected_attr:
            assert_allclose(getattr(solar, ky), exp[ky], rtol=1e-6)


def test_solar_array(daily_flux_benchmarks, expected_attr):
    """Array checking of solar predictions.

    This checks that the outcome of calculating all the values in the test inputs
    simultaneously using array inputs gives the same answers as the original
    iterated implementation.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = daily_flux_benchmarks
    cal = Calendar(inputs["dates"].to_numpy().astype("datetime64[D]"))

    solar = DailySolarFluxes(
        dates=cal,
        latitude=inputs["lat"].to_numpy(),
        elevation=inputs["elv"].to_numpy(),
        sunshine_fraction=inputs["sf"].to_numpy(),
        temperature=inputs["tc"].to_numpy(),
    )

    for ky in expected_attr:
        assert_allclose(getattr(solar, ky), expected[ky])


def test_solar_array_grid(grid_benchmarks):
    """Array checking of solar predictions for more complex inputs.

    This checks that a gridded dataset works with solar.py
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = grid_benchmarks

    cal = Calendar(inputs.time.values.astype("datetime64[D]"))

    # Duplicate lat and elev to same shape as sf and tc (TODO - avoid this!)
    elev = np.broadcast_to(inputs.elev.data[None, :, :], inputs.sf.data.shape)
    lat = np.broadcast_to(inputs.lat.data[None, :, None], inputs.sf.data.shape)

    solar = DailySolarFluxes(
        latitude=lat,
        elevation=elev,
        dates=cal,
        sunshine_fraction=inputs["sf"].data,
        temperature=inputs["tmp"].data,
    )

    # Test that the resulting solar calculations are the same.
    for src_key, solar_key in (
        ("ppfd_d", "daily_ppfd"),
        ("rn_d", "daytime_net_radiation"),
        ("rnn_d", "nighttime_net_radiation"),
    ):
        assert_allclose(
            getattr(solar, solar_key), expected[src_key].data, equal_nan=True, rtol=1e-6
        )
