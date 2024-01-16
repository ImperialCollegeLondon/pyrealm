"""This module tests the main methods in the SplashModel.

- __init__, which initializes the model
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
def splash_model(splash_core_constants, grid_benchmarks):
    """Create a SplashModel object for testing."""

    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, _ = grid_benchmarks

    return SplashModel(
        lat=np.broadcast_to(inputs.lat.data[None, :, None], inputs.sf.data.shape),
        elv=np.broadcast_to(inputs.elev.data[None, :, :], inputs.sf.data.shape),
        dates=Calendar(inputs.time.data),
        sf=inputs.sf.data,
        tc=inputs.tmp.data,
        pn=inputs.pre.data,
        core_const=splash_core_constants,
    )


def test_splash_model_initialization(splash_model, grid_benchmarks):
    """Test the initialization of the SplashModel class."""

    inputs, _ = grid_benchmarks
    assert splash_model.shape == inputs.time.shape + inputs.elev.shape
    assert splash_model.pa.shape == inputs.elev.shape


def test_estimate_daily_water_balance(splash_model):
    """Test the estimate_daily_water_balance method of the SplashModel class."""
    wn_init = np.array([75])
    wn, rn = splash_model.estimate_daily_water_balance(wn_init)

    assert isinstance(wn, np.ndarray)
    assert wn.shape == wn_init.shape
    assert isinstance(rn, np.ndarray)
    assert rn.shape == wn_init.shape


def test_estimate_initial_soil_moisture(splash_model):
    """Test the estimate_initial_soil_moisture method of the SplashModel class."""
    wn_init = np.array([75])
    max_iter = 10
    max_diff = 1.0
    verbose = False

    wn = splash_model.estimate_initial_soil_moisture(
        wn_init, max_iter, max_diff, verbose
    )

    assert isinstance(wn, np.ndarray)
    assert wn.shape == wn_init.shape


def test_calc_soil_moisture(splash_model):
    """Test the calc_soil_moisture method of the SplashModel class."""
    wn_init = np.array([75])

    wn, rn = splash_model.calc_soil_moisture(wn_init)

    assert isinstance(wn, np.ndarray)
    assert wn.shape == wn_init.shape
    assert isinstance(rn, np.ndarray)
    assert rn.shape == wn_init.shape
