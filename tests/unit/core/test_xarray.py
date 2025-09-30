"""Tests for the xarray_inputs decorator and some functions using it."""

import numpy as np
import pytest
import xarray as xr

from pyrealm.core.xarray import xarray_inputs, xarray_inputs_kw


@pytest.fixture
def dataset() -> xr.Dataset:
    """Fixture to load the pmodel_global dataset for testing inputs."""

    from importlib import resources

    file_path = resources.files("pyrealm_build_data.rpmodel") / "pmodel_global.nc"
    with resources.as_file(file_path) as path:
        dataset = xr.open_dataset(path, engine="netcdf4")

    return dataset.sel(latitude=slice(-40, 40))


def test_xarray_inputs():
    """Test xarray_inputs correctly converts xr.DataArrays into np.arrays."""

    # Single input
    in_a = xr.DataArray(np.array([1, 2, 3]))
    out_a = xarray_inputs(in_a)
    assert isinstance(out_a, np.ndarray)
    np.testing.assert_array_equal(out_a, in_a.values)

    # Multiple inputs
    in_a = xr.DataArray(np.array([1, 2, 3]))
    in_b = xr.DataArray(np.ones((2, 3)))
    out_a, out_b = xarray_inputs(in_a, in_b)
    assert isinstance(out_a, np.ndarray)
    assert isinstance(out_b, np.ndarray)
    np.testing.assert_array_equal(out_a.ravel(), in_a.values.ravel())
    np.testing.assert_array_equal(out_b.ravel(), in_b.values.ravel())

    # No inputs
    out_none = xarray_inputs()
    assert out_none == ()

    # No DataArray inputs - check inputs unchanged
    in_a = np.ones((2, 2))
    in_b = np.ones(3)
    out_a, out_b = xarray_inputs(in_a, in_b)
    np.testing.assert_array_equal(out_a, in_a)
    np.testing.assert_array_equal(out_b, in_b)


def test_xarray_inputs_kw():
    """Test xarray_inputs_kw correctly converts xr.DataArrays into np.arrays."""

    # Single input
    in_a = xr.DataArray(np.array([1, 2, 3]))
    (kwargs,) = xarray_inputs_kw(a=in_a)
    out_a = kwargs["a"]
    assert isinstance(out_a, np.ndarray)
    np.testing.assert_array_equal(out_a, in_a.values)

    # Multiple inputs
    in_a = xr.DataArray(np.array([1, 2, 3]))
    in_b = xr.DataArray(np.ones((2, 3)))
    in_c = xr.DataArray(np.ones(1))
    in_d = xr.DataArray(np.ones(1))
    out_a, out_b, kwargs = xarray_inputs_kw(in_a, in_b, c=in_c, d=in_d)
    out_c = kwargs["c"]
    out_d = kwargs["d"]
    assert isinstance(out_a, np.ndarray)
    assert isinstance(out_b, np.ndarray)
    assert isinstance(out_c, np.ndarray)
    assert isinstance(out_d, np.ndarray)
    np.testing.assert_array_equal(out_a.ravel(), in_a.values.ravel())
    np.testing.assert_array_equal(out_b.ravel(), in_b.values.ravel())
    np.testing.assert_array_equal(out_c.ravel(), in_c.values.ravel())
    np.testing.assert_array_equal(out_d.ravel(), in_d.values.ravel())


def test_xarray_inputs_dimensions():
    """Test xarray_inputs correctly expands missing dimensions."""

    in_a = xr.DataArray(np.ones((2, 3)), dims=["1", "2"])
    in_b = xr.DataArray(np.ones(3), dims=["2"])
    in_c = xr.DataArray(np.ones((4, 2)), dims=["3", "1"])

    # xarray_inputs
    out_a, out_b, out_c = xarray_inputs(in_a, in_b, in_c)
    assert isinstance(out_a, np.ndarray)
    assert isinstance(out_b, np.ndarray)
    assert isinstance(out_c, np.ndarray)
    assert out_a.shape == (2, 3, 1)
    assert out_b.shape == (1, 3, 1)
    assert out_c.shape == (2, 1, 4)

    # xarray_inputs_kw
    out_a, out_b, out_c_dict = xarray_inputs_kw(in_a, in_b, c=in_c)
    out_c = out_c_dict["c"]
    assert isinstance(out_a, np.ndarray)
    assert isinstance(out_b, np.ndarray)
    assert isinstance(out_c, np.ndarray)
    assert out_a.shape == (2, 3, 1)
    assert out_b.shape == (1, 3, 1)
    assert out_c.shape == (2, 1, 4)


def test_xarray_pmodel_environment(dataset: xr.Dataset):
    """Test PModelEnvironment can be initialised using xarray inputs without issue."""
    from pyrealm.core.pressure import calc_patm
    from pyrealm.pmodel import PModelEnvironment

    tc = dataset["temp"]
    vpd = dataset["VPD"]
    co2 = dataset["CO2"]
    patm = calc_patm(dataset["elevation"].isel(Time=0))
    PModelEnvironment(tc=tc, vpd=vpd, co2=co2, patm=patm)
