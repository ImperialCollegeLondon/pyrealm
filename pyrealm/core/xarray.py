"""Utilities for handling xarray inputs to functions that expect arrays."""

from collections.abc import Hashable
from typing import TypeVar, overload

import numpy as np
import xarray as xr
from numpy.typing import NDArray

ArrayTypeVar = TypeVar("ArrayTypeVar", NDArray[np.floating], xr.DataArray)
"""Either a :class:`numpy.NDArray` or :class:`xarray.DataArray`.

This is used to define functions that can accept and then return
:class:`xarray.DataArray`.
"""

ArrayType = NDArray[np.floating] | xr.DataArray


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


@overload
def xarray_inputs(array1: NDArray[T] | xr.DataArray, /) -> NDArray[T]: ...
@overload
def xarray_inputs(
    array1: NDArray[T] | xr.DataArray,
    array2: NDArray[T] | xr.DataArray,
    /,
    *other_arrays: NDArray[T] | xr.DataArray,
) -> tuple[NDArray[T], ...]: ...


def xarray_inputs(
    *arrays: NDArray[T] | xr.DataArray,
) -> NDArray[T] | tuple[NDArray[T], ...]:
    """Converts any `xarray.DataArray` inputs to numpy arrays.

    This allows functions that expect numpy arrays to be used directly with
    xarray DataArrays, simplifying compatibility between data types.

    All DataArray inputs will be expanded to have the same set of dimensions. Where
    the expanded dimensions will have a length of one. Note that the order of dimensions
    will depend on the order of inputs - the first input will initially define the order
    and additional dimensions in later inputs will be appended.

    No checking of shape consistency is performed - use check_input_shapes for this.

    Args:
        *arrays: The variables to convert into numpy arrays.

    Returns:
        The stripped array(s).

    Examples:
        >>> input = xr.DataArray([1, 2, 3])
        >>> array = xarray_inputs(input)
        >>> type(array)
        <class 'numpy.ndarray'>
        >>> array
        array([1, 2, 3])
    """

    if len(arrays) == 1:
        a = arrays[0]
        return a.to_numpy() if isinstance(a, xr.DataArray) else a

    else:
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
