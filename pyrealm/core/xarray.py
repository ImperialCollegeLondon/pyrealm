"""Utilities for handling xarray inputs to functions that expect arrays."""

from collections.abc import Hashable
from typing import TypeVar

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def _get_dims(*args: NDArray | xr.DataArray) -> list[Hashable]:
    """Get the full list of dimensions across all DataArray arguments."""
    dims = []
    for arg in args:
        if isinstance(arg, xr.DataArray):
            dims.extend([d for d in arg.dims if d not in dims])
    return dims


def _convert_arg(da: xr.DataArray, dims: list[Hashable]) -> NDArray:
    """Expand DataArray to include all the dimensions and return the numpy array."""
    # Ensure the DataArray includes all of the dimensions in the correct order
    missing = {d: 1 for d in dims if d not in da.dims}
    if missing:
        da = da.expand_dims(missing)
    da = da.transpose(*dims)
    # Convert DataArrays to numpy arrays
    return da.to_numpy()


T = TypeVar("T", bound=np.generic)


def xarray_inputs(*arrays: NDArray[T] | xr.DataArray) -> tuple[NDArray[T], ...]:
    """Converts any `xarray.DataArray` inputs to numpy arrays.

    This allows functions that expect numpy arrays to be used directly with
    xarray DataArrays, simplifying compatibility between data types.

    All DataArray inputs will be expanded to have the same set of dimensions. Where
    the expanded dimensions will have a length of one. Note that the order of dimensions
    will depend on the order of inputs - the first input will initially define the order
    and additional dimensions in later inputs will be appended.

    No checking of shape consistency is performed - use check_input_shapes for this.
    """

    dims = _get_dims(*arrays)
    return tuple(
        _convert_arg(a, dims) if isinstance(a, xr.DataArray) else a for a in arrays
    )


def xarray_inputs_kw(
    *arrays: NDArray[T] | xr.DataArray,
    **kwargs: NDArray[T] | xr.DataArray,
) -> tuple[*tuple[NDArray[T], ...], dict[str, NDArray[T]]]:
    """Converts any `xarray.DataArray` inputs to numpy arrays.

    Performs the same functionality as :func:`xarray_inputs` but can also take - and
    return - a kwargs dictionary.
    """

    dims = _get_dims(*arrays, *kwargs.values())
    new_arrays = tuple(
        _convert_arg(a, dims) if isinstance(a, xr.DataArray) else a for a in arrays
    )
    new_kwargs = {
        k: _convert_arg(v, dims) if isinstance(v, xr.DataArray) else v
        for k, v in kwargs.items()
    }
    return (*new_arrays, new_kwargs)
