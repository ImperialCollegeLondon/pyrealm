"""pytest configuration for the splash submodule."""

from importlib import resources

import numpy as np
import pytest
import xarray


@pytest.fixture()
def daily_flux_benchmarks() -> tuple[np.ndarray, np.ndarray]:
    """Test daily values.

    Loads an input file and SPLASH outputs for 100 random locations with a wide range of
    possible input values. Not intended for testing time series iteration, just the
    daily predictions of all core variables.
    """

    dpath = resources.files("pyrealm_build_data.splash")

    inputs = np.genfromtxt(
        (dpath / "inputs.csv").name,
        dtype=None,
        delimiter=",",
        names=True,
        encoding="UTF-8",
    )

    expected = np.genfromtxt(
        (dpath / "benchmark_daily_fluxes.csv").name,
        dtype=None,
        delimiter=",",
        names=True,
        encoding="UTF-8",
    )

    # rename a couple of fields to match new implementation
    assert expected.dtype.names is not None
    exp_fields = list(expected.dtype.names)
    exp_fields[exp_fields.index("my_nu")] = "nu"
    exp_fields[exp_fields.index("my_lambda")] = "lambda_"
    expected.dtype.names = tuple(exp_fields)

    return inputs, expected


@pytest.fixture()
def one_d_benchmark() -> tuple[xarray.Dataset, xarray.Dataset]:
    """Test one dimensional time series.

    Loads the input data and resulting soil moisture outputs from the single location
    San Francisco dataset provided with the original implementation. These were
    originally calculated using the __main__ code in SPLASH main.py, but the data has
    bee converted to netCDF and run using an alternative interface in order to retain
    more validation data.
    """

    dpath = resources.files("pyrealm_build_data.splash")

    inputs = xarray.load_dataset(dpath / "splash_test_example.nc")

    expected = xarray.load_dataset(dpath / "splash_test_example_out.nc")

    return inputs, expected


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
