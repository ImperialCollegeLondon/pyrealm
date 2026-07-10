"""This module tests the main methods in the SplashModel.

- estimate_daily_water_balance, which estimates soil moisture and run off given
  preceding soil moisture
- estimate_initial_soil_moisture, which assumes a stationary relationship over an
  annual cycle to estimate a starting soil moisture.
- calculate_soil_moisture, which iterates an initial soil moisture forward over a time
  series.


The ``splash_model`` fixture is parameterised using the ``sf_mode`` parameter to switch
between using sunshine fraction and shortwave radiation.
"""

from contextlib import nullcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture(params=[(0, 365), (0, 366), (123, 567)])  # parametrized fixture
def calendar(request, grid_benchmarks):
    """Provide the dates with a random start date."""
    from pyrealm.core.calendar import Calendar

    dates = grid_benchmarks[0].time.to_numpy()
    start, end = request.param
    yield Calendar(dates[start:end])


@pytest.mark.parametrize(
    argnames="flag",
    argvalues=[-1, 1],
)
@pytest.mark.parametrize(argnames="var", argvalues=["lat", "sf", "tmp", "pre", "dates"])
def test_splash_model_init(splash_core_constants, grid_benchmarks, var, flag):
    """Test the initialization of the SplashModel class."""

    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    bounds = dict(
        lat=[-90, 90],
        sf=[0, 1],
        tmp=[-25, 80],
        pre=[0, 1000],
    )

    ds = grid_benchmarks[0].sel(time=slice("2000-01-01", "2000-04-01")).copy()
    dates = ds.time.to_numpy()

    # ensure raising error if calendar size is more or less than timestamps
    if var == "dates":
        if flag < 0:  # less dates than timestamps
            dates = dates[:-1]
        else:  # more dates than timestamps
            ds = ds.sel(time=dates[:-1])
        context = pytest.raises(ValueError)
    # ensure warning if variable is out of bounds
    else:
        vmin, vmax = bounds[var]
        arr = ds[var].to_numpy()
        if flag < 0:  # out of lower bound
            arr.flat[np.random.choice(arr.size)] = vmin - 1e-4
        else:  # out of upper bound
            arr.flat[np.random.choice(arr.size)] = vmax + 1e-4
        context = pytest.warns(UserWarning)

    with context:
        # Convert latitude from (Y) to (1, Y, 1)
        lat = ds.lat.to_numpy()[np.newaxis, :, np.newaxis]
        # Convert elevation from (Y, X) to (1, Y, X)
        elv = ds.elev.to_numpy()[np.newaxis, :, :]

        # Fit the model
        SplashModel(
            latitude=lat,
            elevation=elv,
            dates=Calendar(dates),
            sunshine_fraction=ds.sf.to_numpy(),
            temperature=ds.tmp.to_numpy(),
            precipitation=ds.pre.to_numpy(),
        )


@pytest.fixture
def splash_model(grid_benchmarks, splash_core_constants, calendar, sf_mode):
    """Create a SplashModel object for testing."""

    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.splash import SplashModel

    ds = grid_benchmarks[0].sel(time=calendar.dates)

    latitude = ds.lat.to_numpy()[None, :, None]
    elevation = ds.elev.to_numpy()[None, :, :]
    sunshine_fraction = ds.sf.to_numpy()
    temperature = ds.tmp.to_numpy()
    precipitation = ds.pre.to_numpy()

    # Estimate of a comparable shortwave radiation value for use in testing.
    fluxes = DailySolarFluxes(
        latitude=latitude,
        elevation=elevation,
        sunshine_fraction=sunshine_fraction,
        temperature=temperature,
        dates=calendar,
        core_const=splash_core_constants,
    )
    shortwave_radiation = (
        fluxes.daily_ppfd / splash_core_constants.swdown_to_ppfd_factor
    )

    if sf_mode:
        shortwave_radiation = None
    else:
        sunshine_fraction = None

    splash = SplashModel(
        latitude=latitude,
        elevation=elevation,
        sunshine_fraction=sunshine_fraction,
        shortwave_radiation=shortwave_radiation,
        temperature=temperature,
        precipitation=precipitation,
        dates=calendar,
        core_const=splash_core_constants,
    )

    assert splash.shape == (len(ds.time), *ds.elev.shape)
    return splash


@pytest.mark.parametrize("sf_mode", (True, False))
@pytest.mark.parametrize(
    argnames="overflow,underflow",
    argvalues=[
        pytest.param(0, 0, id="no overflow or underflow"),
        pytest.param(0, 1, id="underflow"),
        pytest.param(1, 0, id="overflow"),
    ],
)
def test_estimate_daily_water_balance(splash_model, sf_mode, overflow, underflow):
    """Test the estimate_daily_water_balance method of the SplashModel class."""

    wn_init = np.random.random(splash_model.shape) * splash_model.kWm
    context = nullcontext()

    if overflow:
        wn_init.flat[np.random.choice(wn_init.size)] = splash_model.kWm.item() + 1e-4
        context = pytest.raises(ValueError)
    if underflow:
        wn_init.flat[np.random.choice(wn_init.size)] = -1e-4
        context = pytest.raises(ValueError)

    with context:
        aet, wn, ro = splash_model.estimate_daily_water_balance(wn_init)
        assert_allclose(
            aet + wn + ro,
            wn_init + splash_model.precipitation + splash_model.evap.cond,
            equal_nan=True,
        )


@pytest.mark.parametrize("sf_mode", (True, False))
def test_estimate_initial_soil_moisture(splash_model):
    """Test the estimate_initial_soil_moisture method of the SplashModel class.

    Convergence currently fails with sf_mode=False (using shortwave radiation)
    """

    if splash_model.shape[0] > 365:
        context = nullcontext()
    else:
        context = pytest.raises(ValueError)

    wn_init = np.random.random(splash_model.shape[1:]) * splash_model.kWm

    with context:
        splash_model.estimate_initial_soil_moisture(
            wn_init, max_iter=100, max_diff=1e-4
        )  # simply check convergence


@pytest.mark.parametrize("sf_mode", (True, False))
def test_calculate_soil_moisture(splash_model, sf_mode):
    """Test the calculate_soil_moisture method of the SplashModel class."""

    wn_init = np.random.random(splash_model.shape[1:]) * splash_model.kWm

    aet, wn, ro = splash_model.calculate_soil_moisture(wn_init)
    wn_prev = np.insert(wn, 0, wn_init, axis=0)[:-1]

    assert_allclose(
        aet + wn + ro,
        wn_prev + splash_model.precipitation + splash_model.evap.cond,
        equal_nan=True,
        rtol=1e-6,
    )


def test_calculate_soil_moisture_1D(calendar):
    """Test for correct assignment in calculate_soil_moisture method for 1D inputs."""

    from pyrealm.splash.splash import SplashModel

    n_days = calendar.n_dates

    splash = SplashModel(
        latitude=np.zeros(n_days),
        elevation=np.zeros(n_days),
        sunshine_fraction=np.zeros(n_days),
        temperature=np.zeros(n_days),
        precipitation=np.zeros(n_days),
        dates=calendar,
    )

    # Check there is no error for (1D) scalar initial value
    wn_init = np.array([1])
    aet, wn, ro = splash.calculate_soil_moisture(wn_init)

    assert aet.shape == (n_days,)
    assert wn.shape == (n_days,)
    assert ro.shape == (n_days,)
