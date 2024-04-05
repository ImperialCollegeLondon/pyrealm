"""Profiling test class.

Runs a profiler on the splash model implementation to identify runtime bottlenecks.
"""

from importlib import resources

import numpy as np
import pytest
import xarray

from pyrealm.core.calendar import Calendar
from pyrealm.splash.splash import SplashModel


@pytest.fixture
def splash_core_constants():
    """Provide constants using SPLASH original defaults.

    SPLASH v1 uses 15°C / 288.1 5 K in the standard atmosphere definition and uses the
    Chen method for calculating water density.
    """
    from pyrealm.constants import CoreConst

    return CoreConst(k_To=288.15, water_density_method="chen")


@pytest.mark.profiling
class TestClass:
    """Test class for the profiler running on the pmodel implementation."""

    @pytest.fixture()
    def setup(self, splash_core_constants):
        """Set up the the splash model and load the test data set."""
        dpath = resources.files("pyrealm_build_data.splash")
        data = xarray.load_dataset(dpath / "data/splash_nw_us_grid_data.nc")

        self.splash = SplashModel(
            lat=np.broadcast_to(data.lat.data[None, :, None], data.sf.data.shape),
            elv=np.broadcast_to(data.elev.data[None, :, :], data.sf.data.shape),
            dates=Calendar(data.time.data),
            sf=data.sf.data,
            tc=data.tmp.data,
            pn=data.pre.data,
            core_const=splash_core_constants,
        )

    def test_profiling_calculate_soil_moisture(self, setup):
        """Profile the calculate_soil_moisture method of the splash model."""
        init_soil_moisture = self.splash.estimate_initial_soil_moisture(verbose=False)
        aet_out, wn_out, ro_out = self.splash.calculate_soil_moisture(
            init_soil_moisture
        )
