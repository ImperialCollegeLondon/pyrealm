"""The bounds_checker module.

This module contains utility functions to handle bounds checking on the inputs to
functions and methods in the :mod:`pyrealm` package. Some functions in :mod:`pyrealm`
are only well-behaved with given bounds and bounds checking also provides a (partial)
check on the units being provided.

As originally implemented, this module provided a subclass of numpy.ma.core.MaskedArray.
The intention was that a constrained array becomes a thing that carries with it a
description of its constraints, which sounds appealing but...

1) MaskedArrays have a performance hit.
2) Subclasses are 'contagious': so x * subclass returns an object of class subclass.
   That then requires the subclass to be extended to handle all required methods.
3) NaN handling. Numpy has np.nan - and masked arrays do handle _masked_ np.nan values,
   but not unmasked ones:

  >>> import numpy as np
  >>> x = np.ma.masked_array([1,2,np.nan, 4], mask=[1,0,0,1])
  >>> x.mean()
  nan
  >>> x = np.ma.masked_array([1,2,np.nan, 4], mask=[1,0,1,0])
  >>> x.mean()
  3.0

So the general approach here is now:

- a bounds checker function that tests if data exceeds bounds and warns when it does.
- a bounds mask function that returns np.ndarrays masked using np.nan when values are
  out of the applied bounds. Because np.nan is a _float_ value, masking is always
  applied to a float version of the input.


Note that this means that the codebase then needs to systematically use nan-aware
functions and possibly using bottleneck for speed.

Part of the problem here is that Numpy lacks a general solution to missing data:
  - https://numpy.org/neps/nep-0026-missing-data-summary
"""

import numpy as np
from numpy.typing import NDArray

from pyrealm import warnings


def _get_interval_functions(interval_type: str = "[]") -> tuple[np.ufunc, np.ufunc]:
    """Converts interval notation type to appropriate functions.

    The interval type should be one of ``[]``, ``()``, ``[)`` or ``(]``. The function
    returns a two tuple of ``numpy.ufunc`` functions that implement the appropriate
    lower and upper boundaries given the interval type.

    Args:
        interval_type: A string describing the interval bounds
    """

    # Implement the interval type
    if interval_type not in ["[]", "()", "[)", "(]"]:
        raise ValueError(f"Unknown interval type: {interval_type}")

    # Default open interval traps values less or greater than
    lower_func: np.ufunc = np.less
    upper_func: np.ufunc = np.greater

    # Closed intervals replace functions with less_eq/greater_wq
    if interval_type[0] == "(":
        lower_func = np.less_equal

    if interval_type[1] == ")":
        upper_func = np.greater_equal

    return lower_func, upper_func


def bounds_checker(
    values: NDArray,
    lower: float = -np.infty,
    upper: float = np.infty,
    interval_type: str = "[]",
    label: str = "",
    unit: str = "",
) -> NDArray:
    r"""Check inputs fall within bounds.

    This is a simple pass through function that tests whether the values fall within
    the bounds specified and issues a warning when this is not the case

    Args:
        values: An np.ndarray
        lower: The value of the lower constraint
        upper: The value of the upper constraint
        interval_type: The interval type of the constraint ('[]', '()', '[)', '(]')
        label: A string giving a descriptive label of the variable for use in warnings.
        unit: A string specifying the expected units.

    Returns:
        The function returns the contents of values.

    Examples:
        >>> vals = np.array([-15, 20, 30, 124], dtype=np.float)
        >>> vals_c = bounds_checker(vals, 0, 100, label='temperature', unit='°C')
    """

    # Get the interval functions
    lower_func, upper_func = _get_interval_functions(interval_type)

    # Do the input values contain out of bound values?
    out_of_bounds = np.logical_or(lower_func(values, lower), upper_func(values, upper))

    if out_of_bounds.sum():
        warnings.warn(
            f"Variable {label} ({unit}) contains values outside "
            f"the expected range ({lower},{upper}). Check units?"
        )

    return values


def bounds_mask(
    inputs: NDArray,
    lower: float = -np.infty,
    upper: float = np.infty,
    interval_type: str = "[]",
    label: str = "",
) -> NDArray[np.floating]:
    r"""Mask inputs that do not fall within bounds.

    This function constrains the values in inputs, replacing values outside the provided
    interval with np.nan. Because np.nan is only defined for float types, the function
    will always return a float array.

    Args:
        inputs: An np.ndarray
        lower: The value of the lower constraint
        upper: The value of the upper constraint
        interval_type: The interval type of the constraint ('[]', '()', '[)', '(]')
        label: A string giving a descriptive label of the constrained contents
            used in reporting.

    Returns:
        If no data is out of bounds, the original inputs are returned, otherwise
        a float np.ndarray object with out of bounds values replaced with np.nan.

    Examples:
        >>> vals = np.array([-15, 20, 30, 124], dtype=np.float)
        >>> np.nansum(vals)
        159.0
        >>> vals_c = input_mask(vals, 0, 100, label='temperature')
        >>> np.nansum(vals_c)
        50.0
        >>> vals_c = input_mask(vals, 0, 124, interval_type='[]', label='temperature')
        >>> np.nansum(vals_c)
        174.0
        >>> vals_c = input_mask(vals, 0, 124, interval_type='[)', label='temperature')
        >>> np.nansum(vals_c)
        50.0
    """

    # Get the interval functions
    lower_func, upper_func = _get_interval_functions(interval_type)

    # Raise error for non-array input
    if not isinstance(inputs, np.ndarray):
        raise TypeError(f"Cannot set bounds on {type(inputs)}")

    # Do the input values contain out of bound values?
    mask = np.logical_or(lower_func(inputs, lower), upper_func(inputs, upper))

    # Check if any masking needs to be done
    if not mask.sum():
        return inputs

    # If an ndarray, then we need a float version to set np.nan and we copy to avoid
    # modifying the original input. Using type
    if not np.issubdtype(inputs.dtype, np.floating):
        # Copies implicitly
        outputs = inputs.astype(np.float32)
    else:
        outputs = inputs.copy()

    # Count the existing number of NaN values - impossible to have nan in an integer
    # input but isnan works with any input.
    initial_na_count = np.isnan(inputs).sum()

    # Fill in np.nan where values around outside constraints
    outputs[mask] = np.nan

    final_na_count = np.isnan(outputs).sum()

    # Report
    warnings.warn(
        f"{final_na_count - initial_na_count} values set to NaN "
        f"using {interval_type[0]}{lower}, {upper}{interval_type[1]} "
        f"bounds on {label}",
        category=RuntimeWarning,
    )

    return outputs
