"""This module tests the main methods in the SplashModel.

- estimate_daily_water_balance, which estimates soil moisture and run off given
  preceeding soil moisture
- estimate_initial_soil_moisture, which assumes a stationary relationship over an
  annual cycle to estimate a starting soil moisture.
- calculate_soil_moisture, which iterates an initial soil moisture forward over a time
  series.
"""

from contextlib import nullcontext

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(params=[(0, 365), (0, 366), (123, 567)])  # parametrized fixture
def calendar(request, grid_benchmarks):
    """Provide the dates with a random start date."""
    from pyrealm.core.calendar import Calendar

    dates = grid_benchmarks[0].time.data
    start, end = request.param
    yield Calendar(dates[start:end])  # fmt: skip


@pytest.fixture
def variable_bounds():
    """Provide the variable bounds from the inputs."""
    return dict(
        lat=[-90, 90],
        sf=[0, 1],
        tmp=[-100, 100],
        pre=[0, None],
    )


@pytest.mark.parametrize(
    argnames="flags",
    argvalues=[
        dict(lat=lat, sf=sf, tmp=tmp, pre=pre)
        for lat in [-1, 0, 1]
        for sf in [-1, 0, 1]
        for tmp in [-1, 0, 1]
        for pre in [-1, 0]
    ],  # -1: underflow, 0: no change, 1: overflow
)
def test_splash_model_init(
    splash_core_constants, grid_benchmarks, flags, variable_bounds
):
    """Test the initialization of the SplashModel class."""

    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    dataset = grid_benchmarks[0].sel(time=slice("2000-01-01", "2000-04-01"))

    def update_vals(ds, var):
        """Update the values of the variable."""
        vmin, vmax = variable_bounds[var]
        flag = flags[var]
        if flag == -1:
            return ds.assign({var: xr.full_like(ds[var], vmin - 1e-4)})
        elif flag == 1:
            return ds.assign({var: xr.full_like(ds[var], vmax + 1e-4)})
        else:
            return ds

    ds = dataset
    for var in variable_bounds:
        ds = update_vals(ds, var)

    if all(x == 0 for x in flags.values()):
        context = nullcontext()
    else:
        context = pytest.raises(ValueError)

    with context:
        splash_model = SplashModel(
            lat=np.broadcast_to(ds.lat.data[None, :, None], ds.sf.data.shape),
            elv=np.broadcast_to(ds.elev.data[None, :, :], ds.sf.data.shape),
            dates=Calendar(ds.time.data),
            sf=ds.sf.data,
            tc=ds.tmp.data,
            pn=ds.pre.data,
            core_const=splash_core_constants,
        )
        assert splash_model.shape == (len(ds.time), *ds.elev.shape)


@pytest.fixture
def splash_model(grid_benchmarks, splash_core_constants, calendar):
    """Create a SplashModel object for testing."""

    from pyrealm.splash.splash import SplashModel

    ds = grid_benchmarks[0].sel(time=calendar.dates)

    return SplashModel(
        lat=np.broadcast_to(ds.lat.data[None, :, None], ds.sf.data.shape),
        elv=np.broadcast_to(ds.elev.data[None, :, :], ds.sf.data.shape),
        dates=calendar,
        sf=ds.sf.data,
        tc=ds.tmp.data,
        pn=ds.pre.data,
        core_const=splash_core_constants,
    )


@pytest.mark.parametrize(
    argnames="overflow,underflow",
    argvalues=[
        pytest.param(0, 0, id="no overflow or underflow"),
        pytest.param(0, 1, id="underflow"),
        pytest.param(1, 0, id="overflow"),
        pytest.param(1, 1, id="overflow and underflow"),
    ],
)
def test_estimate_daily_water_balance(splash_model, overflow, underflow):
    """Test the estimate_daily_water_balance method of the SplashModel class."""

    wn_init = np.random.random(splash_model.shape) * splash_model.kWm
    context = nullcontext()

    if overflow:
        wn_init.flat[0] += splash_model.kWm
        context = pytest.raises(ValueError)
    if underflow:
        wn_init.flat[1] -= splash_model.kWm
        context = pytest.raises(ValueError)

    with context:
        aet, wn, ro = splash_model.estimate_daily_water_balance(wn_init)
        assert np.allclose(
            aet + wn + ro,
            wn_init + splash_model.pn + splash_model.evap.cond,
            equal_nan=True,
        )


def test_estimate_initial_soil_moisture(splash_model):
    """Test the estimate_initial_soil_moisture method of the SplashModel class."""

    if splash_model.shape[0] > 365:
        context = nullcontext()
    else:
        context = pytest.raises(ValueError)

    wn_init = np.random.random(splash_model.shape[1:]) * splash_model.kWm

    with context:
        splash_model.estimate_initial_soil_moisture(
            wn_init, max_iter=100, max_diff=1e-4
        )  # simply check convergence


def test_calc_soil_moisture(splash_model):
    """Test the calc_soil_moisture method of the SplashModel class."""

    wn_init = np.random.random(splash_model.shape[1:]) * splash_model.kWm

    aet, wn, ro = splash_model.calculate_soil_moisture(wn_init)
    wn_prev = np.insert(wn, 0, wn_init, axis=0)[:-1]

    assert np.allclose(
        aet + wn + ro,
        wn_prev + splash_model.pn + splash_model.evap.cond,
        equal_nan=True,
    )
