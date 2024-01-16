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


def test_splash_model_initialization(splash, grid_benchmarks):
    """Test the initialization of the SplashModel class."""

    inputs, _ = grid_benchmarks
    assert splash.shape == inputs.time.shape + inputs.elev.shape
    assert splash.pa.shape == inputs.elev.shape


def test_estimate_daily_water_balance(splash):
    """Test the estimate_daily_water_balance method of the SplashModel class."""


def test_estimate_initial_soil_moisture(splash):
    """Test the estimate_initial_soil_moisture method of the SplashModel class."""


def test_calc_soil_moisture():
    """Test the calc_soil_moisture method of the SplashModel class."""
