"""Tests for the xarray_inputs decorator and some functions using it."""

import numpy as np
import pytest
import xarray as xr

from pyrealm.core.xarray import (
    get_common_dims,
    is_arraytype,
    xarray_inputs,
)


@pytest.mark.parametrize(
    "value, result",
    [
        pytest.param(np.array([1, 2]), True, id="np.array"),
        pytest.param(xr.DataArray([1, 2]), True, id="xr.DataArray"),
        pytest.param([1, 2], False, id="list"),
        pytest.param(2, False, id="scalar"),
        pytest.param({"a": 2}, False, id="dict"),
        pytest.param(None, False, id="None"),
    ],
)
def test_is_arraytype(value, result):
    """Check is_arraytype correctly identifies if objects are `ArrayType`s."""
    assert is_arraytype(value) is result


def test_get_common_dims():
    """Test get_common_dims gives expected outputs."""
    input_1 = xr.DataArray([], dims="a")
    input_2 = xr.DataArray([[]], dims=["b", "a"])
    input_3 = np.array([])
    dims = get_common_dims(input_1, input_2, input_3)
    assert dims == ["a", "b"]


@pytest.fixture
def dataset() -> xr.Dataset:
    """Fixture to load the pmodel_global dataset for testing inputs."""

    from importlib import resources

    file_path = resources.files("pyrealm_build_data.rpmodel") / "pmodel_global.nc"
    with resources.as_file(file_path) as path:
        dataset = xr.open_dataset(path, engine="netcdf4")

    return dataset.sel(latitude=slice(-40, 40))


def test_xarray_inputs_single():
    """Test xarray_inputs converts a single xr.DataArray into np.array."""
    in_a = xr.DataArray(np.array([1, 2, 3]))
    out_a = xarray_inputs(in_a)
    assert isinstance(out_a, np.ndarray)
    np.testing.assert_array_equal(out_a, in_a.values)


def test_xarray_inputs_multiple():
    """Test xarray_inputs converts multiple xr.DataArrays into np.arrays."""
    in_a = xr.DataArray(np.array([1, 2, 3]))
    in_b = xr.DataArray(np.ones((2, 3)))
    out_a, out_b = xarray_inputs(in_a, in_b)
    assert isinstance(out_a, np.ndarray)
    assert isinstance(out_b, np.ndarray)
    np.testing.assert_array_equal(out_a.ravel(), in_a.values.ravel())
    np.testing.assert_array_equal(out_b.ravel(), in_b.values.ravel())


def test_xarray_inputs_empty():
    """Test xarray_inputs outputs an empty tuple if there are no inputs."""
    out_none = xarray_inputs()
    assert out_none == ()


def test_xarray_inputs_numpy():
    """Test xarray_inputs leaves the inputs unchanged if just np.array inputs."""
    in_a = np.ones((2, 2))
    in_b = np.ones(3)
    out_a, out_b = xarray_inputs(in_a, in_b)
    np.testing.assert_array_equal(out_a, in_a)
    np.testing.assert_array_equal(out_b, in_b)


def test_xarray_inputs_mixed():
    """Test xarray_inputs for a mix of np.arrays and xr.DataArrays inputs."""
    in_a = xr.DataArray(np.ones((2, 3)))
    in_b = np.ones(3)
    out_a, out_b = xarray_inputs(in_a, in_b)
    assert isinstance(out_a, np.ndarray)
    np.testing.assert_array_equal(out_a, in_a.values)
    np.testing.assert_array_equal(out_b, in_b)


def test_xarray_inputs_kw_single():
    """Test xarray_inputs converts a single xr.DataArray into np.array."""
    in_a = xr.DataArray(np.array([1, 2, 3]))
    _, kwargs = xarray_inputs(kwargs={"a": in_a})
    out_a = kwargs["a"]
    assert isinstance(out_a, np.ndarray)
    np.testing.assert_array_equal(out_a, in_a.values)


def test_xarray_inputs_kw_multiple():
    """Test xarray_inputs converts multiple xr.DataArrays into np.arrays."""
    in_a = xr.DataArray(np.array([1, 2, 3]))
    in_b = xr.DataArray(np.ones((2, 3)))
    in_c = xr.DataArray(np.ones(1))
    in_d = xr.DataArray(np.ones(1))
    (out_a, out_b), kwargs = xarray_inputs(in_a, in_b, kwargs={"c": in_c, "d": in_d})
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


@pytest.mark.parametrize("use_kwargs", [True, False])
@pytest.mark.parametrize("dims", [None, ["1", "2", "3", "4"]])
def test_xarray_inputs_dimensions(use_kwargs, dims):
    """Test xarray_inputs correctly expands missing dimensions."""

    in_a = xr.DataArray(np.ones((2, 3)), dims=["1", "2"])
    in_b = xr.DataArray(np.ones(3), dims=["2"])
    in_c = xr.DataArray(np.ones((4, 2)), dims=["3", "1"])

    if not use_kwargs:
        out_a, out_b, out_c = xarray_inputs(in_a, in_b, in_c, dims=dims)
    else:
        arrays, kw_arrays = xarray_inputs(in_a, in_b, kwargs={"c": in_c}, dims=dims)
        out_a, out_b = arrays
        out_c = kw_arrays["c"]

    assert isinstance(out_a, np.ndarray)
    assert isinstance(out_b, np.ndarray)
    assert isinstance(out_c, np.ndarray)
    if dims is None:
        assert out_a.shape == (2, 3, 1)
        assert out_b.shape == (1, 3, 1)
        assert out_c.shape == (2, 1, 4)
    else:
        assert out_a.shape == (2, 3, 1, 1)
        assert out_b.shape == (1, 3, 1, 1)
        assert out_c.shape == (2, 1, 4, 1)


def test_xarray_pmodel_environment(dataset: xr.Dataset):
    """Test PModelEnvironment can be initialised using xarray inputs without issue."""
    from pyrealm.core.pressure import calc_patm
    from pyrealm.pmodel import PModelEnvironment

    tc = dataset["temp"]
    vpd = dataset["VPD"]
    co2 = dataset["CO2"]
    patm = calc_patm(dataset["elevation"].isel(Time=0))
    PModelEnvironment(tc=tc, vpd=vpd, co2=co2, patm=patm)
