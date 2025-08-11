"""Tests for the xarray_inputs decorator and some functions using it."""

import numpy as np
import pytest
import xarray as xr

from pyrealm.core.xarray import xarray_inputs


@pytest.fixture
def dataset():
    """Fixture to load the pmodel_global dataset for testing inputs."""

    from importlib import resources

    file_path = resources.files("pyrealm_build_data.rpmodel") / "pmodel_global.nc"
    with resources.as_file(file_path) as path:
        dataset = xr.open_dataset(path, engine="netcdf4")

    return dataset.sel(latitude=slice(-40, 40))


@xarray_inputs
def dummy_func(*args):
    """Dummy function that just returns arguments for inspection."""
    return args


def test_xarray_inputs_decorator():
    """Test the xarray_inputs decorator correctly converts xarray inputs."""

    # Single input
    in_a = xr.DataArray(np.array([1, 2, 3]))
    (out_a,) = dummy_func(in_a)
    assert isinstance(out_a, np.ndarray)
    np.testing.assert_array_equal(out_a, in_a.values)

    # Multiple inputs
    in_a = xr.DataArray(np.array([1, 2, 3]))
    in_b = "string"
    in_c = xr.DataArray(np.ones((2, 3)))
    out_a, out_b, out_c = dummy_func(in_a, in_b, in_c)
    assert isinstance(out_a, np.ndarray)
    assert isinstance(out_b, str)
    assert isinstance(out_c, np.ndarray)
    np.testing.assert_array_equal(out_a, in_a.values)
    np.testing.assert_array_equal(out_c, in_c.values)


def test_xarray_pmodel_environment(dataset):
    """Test PModelEnvironment can be initialised using xarray inputs without issue."""
    from pyrealm.core.pressure import calc_patm
    from pyrealm.pmodel import PModelEnvironment

    tc = dataset["temp"]
    vpd = dataset["VPD"]
    co2 = dataset["CO2"]
    patm = calc_patm(dataset["elevation"].isel(Time=0))
    PModelEnvironment(tc=tc, vpd=vpd, co2=co2, patm=patm)
