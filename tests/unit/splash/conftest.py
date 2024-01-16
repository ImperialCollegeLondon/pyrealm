"""pytest configuration for the splash submodule."""

from importlib import resources

import pytest
import xarray


@pytest.fixture
def splash_core_constants():
    """Provide constants using SPLASH original defaults.

    SPLASH v1 uses 15°C / 288.1 5 K in the standard atmosphere definition and uses the
    Chen method for calculating water density.
    """

    from pyrealm.constants import CoreConst

    return CoreConst(k_To=288.15, water_density_method="chen")


@pytest.fixture()
def grid_benchmarks() -> tuple[xarray.Dataset, xarray.Dataset]:
    """Test 3D time series.

    This provides a 20 x 20 cell chunk of daily data from WFDE5 v2 and CRU for the west
    coast of the US over two years. It provides a test of iterated calculations and
    checking that applying functions across arrays, not iterating through individual
    time series gives identical results.
    """

    dpath = resources.files("pyrealm_build_data.splash")

    inputs = xarray.load_dataset(dpath / "splash_test_grid_nw_us.nc")

    expected = xarray.load_dataset(dpath / "splash_test_grid_nw_us_out.nc")

    return inputs, expected
