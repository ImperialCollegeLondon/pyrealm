"""Utilities for handling xarray inputs to functions that expect arrays."""

import functools
from collections.abc import Callable
from typing import Any

import xarray as xr


def xarray_inputs(fn: Callable) -> Callable:
    """Decorator that converts any `xarray.DataArray` inputs to numpy arrays.

    This allows functions that expect numpy arrays to be used directly with
    xarray DataArrays, simplifying compatibility between data types.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert xarray inputs to numpy arrays
        args = tuple(a.to_numpy() if isinstance(a, xr.DataArray) else a for a in args)
        kwargs = {
            k: v.to_numpy() if isinstance(v, xr.DataArray) else v
            for k, v in kwargs.items()
        }
        return fn(*args, **kwargs)

    return wrapper
