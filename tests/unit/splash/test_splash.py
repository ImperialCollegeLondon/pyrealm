"""This module tests the main methods in the SplashModel.

- estimate_daily_water_balance, which estimates soil moisture and run off given
  preceeding soil moisture
- estimate_initial_soil_moisture, which assumes a stationary relationship over an
  annual cycle to estimate a starting soil moisture.
- calculate_soil_moisture, which iterates an initial soil moisture forward over a time
  series.
"""

import numpy as np
import pytest


@pytest.fixture
def calendar(grid_benchmarks):
    """Provide the dates with a random start date."""
    from pyrealm.core.calendar import Calendar

    T = 366
    dates = grid_benchmarks[0].time.data
    t0 = np.random.randint(0, len(dates) - T)
    return Calendar(inputs.time.data[t0:t0+T])  # fmt: skip


@pytest.fixture
def inputs(grid_benchmarks, calendar):
    """Provide the inputs from the grid_benchmarks fixture."""
    return grid_benchmarks[0].sel(time=calendar.dates)


@pytest.fixture
def expected(grid_benchmarks, calendar):
    """Provide the expected outputs from the grid_benchmarks fixture."""
    return grid_benchmarks[1].sel(time=calendar.dates)


@pytest.fixture
def splash_model(splash_core_constants, inputs, calendar):
    """Create a SplashModel object for testing."""

    from pyrealm.splash.splash import SplashModel

    splash_model = SplashModel(
        lat=np.broadcast_to(inputs.lat.data[None, :, None], inputs.sf.data.shape),
        elv=np.broadcast_to(inputs.elev.data[None, :, :], inputs.sf.data.shape),
        dates=calendar,
        sf=inputs.sf.data,
        tc=inputs.tmp.data,
        pn=inputs.pre.data,
        core_const=splash_core_constants,
    )

    assert splash_model.shape == calendar.shape + inputs.elev.shape
    return splash_model


def test_estimate_daily_water_balance(splash_model):
    """Test the estimate_daily_water_balance method of the SplashModel class."""

    wn_init = np.random.random(splash_model.shape) * splash_model.kWm
    aet, wn, rn = splash_model.estimate_daily_water_balance(wn_init)

    assert np.allclose(
        aet + wn + rn, wn_init + splash_model.pn + splash_model.evap.cond
    )


def test_estimate_initial_soil_moisture(splash_model, expected):
    """Test the estimate_initial_soil_moisture method of the SplashModel class."""

    wn_init = np.random.random(splash_model.shape) * splash_model.kWm
    wn = splash_model.estimate_initial_soil_moisture(wn_init)

    assert np.allclose(wn, expected["wn_spun_up"].data)


def test_calc_soil_moisture(splash_model, expected):
    """Test the calc_soil_moisture method of the SplashModel class."""

    aet, wn, ro = splash_model.calculate_soil_moisture(expected["wn_spun_up"].data)

    assert np.allclose(aet, expected["aet_d"].data, equal_nan=True)
    assert np.allclose(wn, expected["wn"].data, equal_nan=True)
    assert np.allclose(ro, expected["ro"].data, equal_nan=True, atol=1e-04)
