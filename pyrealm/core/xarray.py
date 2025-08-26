"""Utilities for handling xarray inputs to functions that expect arrays."""

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr


def _get_dims(args: list[xr.DataArray]) -> list[str]:
    """Get the full list of dimensions across all DataArray arguments."""
    dims = []
    for arg in args:
        if isinstance(arg, xr.DataArray):
            dims.extend([d for d in arg.dims if d not in dims])
    return dims


def _convert_arg(da: xr.DataArray, dims: list[str]) -> np.ndarray:
    """Expand DataArray to include all the dimensions and return the numpy array."""
    # Ensure the DataArray includes all of the dimensions in the correct order
    missing = {d: 1 for d in dims if d not in da.dims}
    if missing:
        da = da.expand_dims(missing)
    da = da.transpose(*dims)
    # Convert DataArrays to numpy arrays
    return da.to_numpy()


def xarray_inputs(fn: Callable) -> Callable:
    """Decorator that converts any `xarray.DataArray` inputs to numpy arrays.

    This allows functions that expect numpy arrays to be used directly with
    xarray DataArrays, simplifying compatibility between data types.

    All DataArray inputs will be expanded to have the same set of dimensions. Where
    the expanded dimensions will have a length of one. Note that the order of dimensions
    will depend on the order of inputs - the first input will initially define the order
    and additional dimensions in later inputs will be appended.

    No checking of shape consistency is performed - use check_input_shapes for this.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get the list of DataArrays (if any)
        inputs = [*args, *kwargs.values()]
        data_arrays = [arg for arg in inputs if isinstance(arg, xr.DataArray)]
        if data_arrays:
            # Get the dimensions
            dims = _get_dims(data_arrays)
            # Convert the DataArray inputs and expand their dimensions
            args = tuple(
                _convert_arg(a, dims) if isinstance(a, xr.DataArray) else a
                for a in args
            )
            kwargs = {
                k: _convert_arg(v, dims) if isinstance(v, xr.DataArray) else v
                for k, v in kwargs.items()
            }
        return fn(*args, **kwargs)

    return wrapper
