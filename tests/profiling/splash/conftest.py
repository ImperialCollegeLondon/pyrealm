"""Some pytest configuration specific to the profiling test suite."""

import gc
from importlib import resources

import numpy as np
import pytest
import xarray


@pytest.fixture(scope="package")
def splash_profile_data(pytestconfig):
    """Setup the data for the profiling.

    This uses a package scoped yield fixture to only load the data once for the splash
    tests and then to delete it all when the tests exit.

    This currently just tiles the data along the longitude axis to increase the load.
    Tiling along the latitude axis adds more complexity - all results at the same
    latitude should be identical, so tiling on longitude is easy.

    Scaling along the years axis is slightly more problematic because the years need to
    be properly calendar aware for splash models and so would need to handle leap year
    sequencing. But a simple fix would be to just calculate the number of days and then
    create the dates as a sequence from the first date - there would be some days that
    change year but it would be very very minor and not really relevant to this test.
    """

    from pyrealm.core.calendar import Calendar

    print("\n***\nSetting up profiling of splash profiling data\n***\n")

    dpath = resources.files("pyrealm_build_data.splash")
    data = xarray.load_dataset(dpath / "data/splash_nw_us_grid_data.nc")

    # Scale up data to give a more robust profile
    def _scale_up(array, scaling_factor: int):
        """Bulk out the data."""
        # Broadcast on lon axis
        return np.tile(array, (1, 1, scaling_factor))

    # Variable set up
    scaling_factor = pytestconfig.getoption("splash_profile_scaleup")

    sf = _scale_up(data.sf.to_numpy(), scaling_factor=scaling_factor)
    tc = _scale_up(data.tmp.to_numpy(), scaling_factor=scaling_factor)
    pn = _scale_up(data.pre.to_numpy(), scaling_factor=scaling_factor)

    elv = np.tile(data.elev.to_numpy(), (1, scaling_factor))
    elv = np.broadcast_to(elv[None, :, :], sf.shape)
    lat = np.broadcast_to(data.lat.to_numpy()[None, :, None], sf.shape)

    dates = Calendar(data.time.data)

    yield sf, tc, pn, elv, lat, dates

    print("\n***\nTearing down\n***\n")

    # Delete all the local objects
    del list(locals().keys())[:]
    gc.collect()
