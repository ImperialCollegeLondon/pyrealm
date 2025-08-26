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
def xarray_inputs_dummy_func(*args, **kwargs):
    """Dummy function that just returns arguments for inspection."""
    return [*args, *kwargs.values()]


def test_xarray_inputs_decorator():
    """Test the xarray_inputs decorator correctly converts xarray inputs."""

    # Single input
    in_a = xr.DataArray(np.array([1, 2, 3]))
    (out_a,) = xarray_inputs_dummy_func(in_a)
    assert isinstance(out_a, np.ndarray)
    np.testing.assert_array_equal(out_a, in_a.values)

    # Multiple inputs
    in_a = xr.DataArray(np.array([1, 2, 3]))
    in_b = "string"
    in_c = xr.DataArray(np.ones((2, 3)))
    out_a, out_b, out_c = xarray_inputs_dummy_func(in_a, in_b, c=in_c)
    assert isinstance(out_a, np.ndarray)
    assert isinstance(out_b, str)
    assert isinstance(out_c, np.ndarray)
    np.testing.assert_array_equal(out_a.ravel(), in_a.values.ravel())
    np.testing.assert_array_equal(out_c.ravel(), in_c.values.ravel())

    # No inputs - check for no error
    xarray_inputs_dummy_func()

    # No DataArray inputs - check inputs unchanged
    in_a = np.ones((2, 2))
    in_b = np.ones(3)
    in_c = "string"
    out_a, out_b, out_c = xarray_inputs_dummy_func(in_a, in_b, in_c)
    np.testing.assert_array_equal(out_a, in_a)
    np.testing.assert_array_equal(out_b, in_b)
    assert out_c == in_c


def test_xarray_inputs_decorator_dimensions():
    """Test the xarray_inputs decorator correctly expands missing dimensions."""

    in_a = xr.DataArray(np.ones((2, 3)), dims=["a", "b"])
    in_b = xr.DataArray(np.ones(3), dims=["b"])
    in_c = xr.DataArray(np.ones((4, 2)), dims=["c", "a"])
    out_a, out_b, out_c = xarray_inputs_dummy_func(in_a, in_b, c=in_c)
    assert isinstance(out_a, np.ndarray)
    assert isinstance(out_b, np.ndarray)
    assert isinstance(out_c, np.ndarray)
    assert out_a.shape == (2, 3, 1)
    assert out_b.shape == (1, 3, 1)
    assert out_c.shape == (2, 1, 4)


def test_xarray_pmodel_environment(dataset):
    """Test PModelEnvironment can be initialised using xarray inputs without issue."""
    from pyrealm.core.pressure import calc_patm
    from pyrealm.pmodel import PModelEnvironment

    tc = dataset["temp"]
    vpd = dataset["VPD"]
    co2 = dataset["CO2"]
    patm = calc_patm(dataset["elevation"].isel(Time=0))
    PModelEnvironment(tc=tc, vpd=vpd, co2=co2, patm=patm)
