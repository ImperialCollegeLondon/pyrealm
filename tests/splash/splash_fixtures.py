from pathlib import Path

import numpy as np
import pytest
import xarray


@pytest.fixture()
def daily_flux_benchmarks(shared_datadir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Test daily values.

    Loads an input file and SPLASH outputs for 100 random locations with a wide range of
    possible input values. Not intended for testing time series iteration, just the
    daily predictions of all core variables."""

    inputs = np.genfromtxt(
        shared_datadir / "inputs.csv",
        dtype=None,
        delimiter=",",
        names=True,
        encoding="UTF-8",
    )

    expected = np.genfromtxt(
        shared_datadir / "benchmark_daily_fluxes.csv",
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
def one_d_benchmark(shared_datadir: Path) -> tuple[xarray.Dataset, xarray.Dataset]:
    """Test one dimensional time series.

    Loads the input data and resulting soil moisture outputs from the single location
    San Francisco dataset provided with the original implementation. These were
    originally calculated using the __main__ code in SPLASH main.py, but the data has
    bee converted to netCDF and run using an alternative interface in order to retain
    more validation data.
    """

    inputs = xarray.load_dataset(shared_datadir / "splash_test_example.nc")

    expected = xarray.load_dataset(shared_datadir / "splash_test_example_out.nc")

    return inputs, expected


@pytest.fixture()
def grid_benchmarks(shared_datadir: Path) -> tuple[xarray.Dataset, xarray.Dataset]:
    """Test 3D time series.

    This provides a 20 x 20 cell chunk of daily data from WFDE5 v2 and CRU for the west
    coast of the US over two years. It provides a test of iterated calculations and
    checking that applying functions across arrays, not iterating through individual
    time series gives identical results."""

    inputs = xarray.load_dataset(shared_datadir / "splash_test_grid_nw_us.nc")

    expected = xarray.load_dataset(shared_datadir / "splash_test_grid_nw_us_out.nc")

    return inputs, expected
