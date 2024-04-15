"""Profiling test class.

Runs a profiler on the splash model implementation to identify runtime bottlenecks.
"""

from importlib import resources

import numpy as np
import pytest
import xarray


@pytest.fixture(scope="module")
def setup():
    """Setup the data for the profiling.

    This currently just tiles the data along the longitude axis to increase the load.
    Tiling along the latitude axis adds more complexity - all results at the same
    latitude should be identical, so tiling on longitude is easy.

    Tiling on time is a pain because of leap years, so even though the year tile is
    there, don't use it. The clever thing to do here is to split those two years into
    leap and non leap and then build the pattern up that way.
    """

    from pyrealm.core.calendar import Calendar

    dpath = resources.files("pyrealm_build_data.splash")
    data = xarray.load_dataset(dpath / "data/splash_nw_us_grid_data.nc")

    # Scale up data to give a more robust profile
    def _scale_up(array, lon_tile=5, year_tile=1):
        """Bulk out the data."""
        # Tile on lon and time axis
        return np.tile(array, (year_tile, 1, lon_tile))

    lon_tile = 500
    year_tile = 1

    sf = _scale_up(data.sf.to_numpy(), lon_tile=lon_tile, year_tile=year_tile)
    tc = _scale_up(data.tmp.to_numpy(), lon_tile=lon_tile, year_tile=year_tile)
    pn = _scale_up(data.pre.to_numpy(), lon_tile=lon_tile, year_tile=year_tile)

    elv = np.tile(data.elev.to_numpy(), (1, lon_tile))
    elv = np.broadcast_to(elv[None, :, :], sf.shape)
    lat = np.broadcast_to(data.lat.to_numpy()[None, :, None], sf.shape)

    dates = Calendar(data.time.data)

    return sf, tc, pn, elv, lat, dates


@pytest.mark.profiling
def test_profile_splash(setup):
    """Run a splash analysis for profiling."""

    from pyrealm.constants import CoreConst
    from pyrealm.splash.splash import SplashModel

    # SPLASH v1 uses 15°C / 288.1 5 K in the standard atmosphere definition and uses the
    # Chen method for calculating water density.

    splash_core_constants = CoreConst(k_To=288.15, water_density_method="chen")

    # Extract the input data
    sf, tc, pn, elv, lat, dates = setup

    # Create the model
    splash = SplashModel(
        lat=lat,
        elv=elv,
        dates=dates,
        sf=sf,
        tc=tc,
        pn=pn,
        core_const=splash_core_constants,
    )

    # Run the initial soil calculation and then the time series.
    init_soil_moisture = splash.estimate_initial_soil_moisture(verbose=False)
    aet_out, wn_out, ro_out = splash.calculate_soil_moisture(init_soil_moisture)
