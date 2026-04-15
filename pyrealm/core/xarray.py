"""Utilities for handling xarray inputs to functions that expect arrays."""

from collections.abc import Hashable
from typing import Any, TypeAlias, TypeGuard, TypeVar, overload

import numpy as np
import xarray as xr
from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic)

# Still using the old-style here because the sphinx cross-references otherwise fail.
# The workaround in `resolve_alias_fallback` docs/source/conf.py requires the old-style.
ArrayType: TypeAlias = NDArray[T] | xr.DataArray  # noqa: UP040
"""Type for array inputs. A union of numpy arrays and xarray DataArrays."""


def is_arraytype(var: Any) -> TypeGuard[ArrayType]:
    """Checks if a variable is an instance of ArrayType."""
    if not isinstance(var, np.ndarray | xr.DataArray):
        return False
    return True


def get_common_dims(
    *arrays: NDArray | xr.DataArray,
    init_dims: list[Hashable] | None = None,
) -> list[Hashable]:
    """Get the full list of dimensions across all DataArray arguments.

    This needs to be called when there are arrays with multiple dtypes that cannot be
    combined in a single call to `xarray_inputs`.

    Args:
        *arrays: The variables to convert into numpy arrays.
        init_dims: An optional list of dims to start with.

    Returns:
        A list of dimension names.

    Examples:
        >>> input_1 = xr.DataArray([], dims="a")
        >>> input_2 = xr.DataArray([[]], dims=["b", "a"])
        >>> get_common_dims(input_1, input_2)
        ['a', 'b']
        >>> get_common_dims(input_1, input_2, init_dims=["c"])
        ['c', 'a', 'b']
    """

    dims = init_dims or []
    for array in arrays:
        if isinstance(array, xr.DataArray):
            dims.extend([d for d in array.dims if d not in dims])
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


# 1 array input
@overload
def xarray_inputs[T: np.generic](
    array1: ArrayType[T],
    /,
    *,
    kwargs: None = ...,
    dims: list[Hashable] | None = ...,
    dims_full: bool = ...,
) -> NDArray[T]: ...


# Multiple array inputs
@overload
def xarray_inputs[T: np.generic](
    array1: ArrayType[T],
    array2: ArrayType[T],
    /,
    *other_arrays: ArrayType[T],
    kwargs: None = ...,
    dims: list[Hashable] | None = ...,
    dims_full: bool = ...,
) -> tuple[NDArray[T], ...]: ...


# Kwargs input
@overload
def xarray_inputs[T: np.generic](
    *arrays: ArrayType[T],
    kwargs: dict[str, ArrayType[T]],
    dims: list[Hashable] | None = ...,
    dims_full: bool = ...,
) -> tuple[tuple[NDArray[T], ...], dict[str, NDArray[T]]]: ...


def xarray_inputs[T: np.generic](
    *arrays: ArrayType[T],
    kwargs: dict[str, ArrayType[T]] | None = None,
    dims: list[Hashable] | None = None,
    dims_full: bool = True,
) -> (
    NDArray[T]
    | tuple[NDArray[T], ...]
    | tuple[tuple[NDArray[T], ...], dict[str, NDArray[T]]]
):
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
        kwargs: An optional dictionary of variables to convert to numpy arrays.
        dims: An optional list of dimension names to expand DataArrays to.
        dims_full: If `dims` is provided this indicates whether this is the full list of
            dimensions (True) or can be added to by `arrays` (default: True).

    Returns:
        The stripped array(s). As a tuple if more than one is provided. As (tuple, dict)
        if kwargs is provided, with the dict containing the converted kwargs.

    Examples:
        >>> input = xr.DataArray([1, 2, 3])
        >>> array = xarray_inputs(input)
        >>> type(array)
        <class 'numpy.ndarray'>
        >>> array
        array([1, 2, 3])
    """

    if dims is None:
        dims = get_common_dims(*arrays, *(kwargs or {}).values())
    elif not dims_full:
        dims = get_common_dims(*arrays, *(kwargs or {}).values(), init_dims=dims)

    # 1 value - return scalar
    if len(arrays) == 1 and not kwargs:
        a = arrays[0]
        return _convert_arg(a, dims) if isinstance(a, xr.DataArray) else a

    else:
        new_arrays = tuple(
            _convert_arg(a, dims) if isinstance(a, xr.DataArray) else a for a in arrays
        )

        # No kwargs - return tuple of arrays
        if kwargs is None:
            return new_arrays

        # kwargs - return tuple including the kwargs dictionary
        else:
            new_kwargs = {
                k: _convert_arg(v, dims) if isinstance(v, xr.DataArray) else v
                for k, v in kwargs.items()
            }
            return (new_arrays, new_kwargs)
