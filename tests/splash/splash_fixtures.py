import numpy as np
import pytest
import xarray


@pytest.fixture()
def daily_flux_benchmarks(shared_datadir):
    """Test values.

    Loads the input file and solar outputs from the original implementation into numpy
    structured arrays"""

    # TODO share this across splash test suite somehow

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
def grid_benchmarks(shared_datadir):
    """Test values.

    Loads the input file and solar outputs from the original implementation into numpy
    structured arrays"""

    # TODO share this across splash test suite somehow

    inputs = xarray.load_dataset(shared_datadir / "splash_test_grid.nc")

    expected = xarray.load_dataset(shared_datadir / "splash_test_grid_out_r1.nc")

    return inputs, expected
