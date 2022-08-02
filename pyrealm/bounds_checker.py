"""The bounds_checker module.

This module contains utility functions to handle bounds checking on the inputs to
functions and methods in the `pyrealm` package. Some functions are only
well-behaved with given bounds and bounds checking also provides a (partial)
check on the units being provided.
"""

from numbers import Number
from typing import Union

import numpy as np

from pyrealm import warnings

# DESIGN NOTES: DO 18/08/21
#
# As originally implemented (still commented below), this module provided a
# subclass of numpy.ma.core.MaskedArray. The intention was that a constrained
# array becomes a thing that carries with it a description of its constraints,
# which sounds appealing but...
#
# 1) MaskedArrays have a performance hit.
# 2) Subclasses are 'contagious': so x * subclass returns an object of class
#    subclass. The attributes of that subclass weren't being set and I suspect
#    that is fixable but it doesn't make sense to do so (unless you were mad
#    enough to implement bounds on _all_ values)
# 3) NaN handling. Numpy has np.nan - and masked arrays do handle _masked_
#    np.nan values, but not unmasked ones:
#
#   x = np.ma.masked_array([1,2,np.nan, 4], mask=[1,0,0,1])
#   x.mean()
#   Out[49]: nan
#   x = np.ma.masked_array([1,2,np.nan, 4], mask=[1,0,1,0])
#   x.mean()
#   Out[51]: 3.0
#
# So the general approach here is now:
#
# - a data constraint function that returns simple np.ndarrays or np.nan
# - uses np.nan for both missing / out of range
# - (maybe) has a public mechanism for setting those constraints, rather than
#   hard coding them. But if the ranges are well documented and sane, then
#   this might be fussy, so parking for now.
#
# This is simpler and cleaner but the codebase then needs to systematically
# use nan-aware functions and possibly using bottleneck for speed. Part of
# the problem here is that Numpy lacks a general solution to missing  data:
#   - https://numpy.org/neps/nep-0026-missing-data-summary
#
# Two issues:
# 1) np.nan is a _float_ value and so this can only be applied to float types
#    and hence some type conversion happens.
# 2) The ability to check for already checked data is lost from the previous
#    implementation. This requires the array object to carry attributes, and
#    that implies subclasses.
#
# NOTES: DO 20/11/2021
#
# Actually imposing constraints and enforcing masking is too heavy handed
# and causes user issues. So, instead retain some of this as a mechanism for
# helping users sanitise inputs, but provide a simple routine to check whether
# values are sane.


def bounds_checker(
    values: Union[np.ndarray, Number],
    lower: Number = -np.infty,
    upper: Number = np.infty,
    interval_type: str = "[]",
    label: str = "",
    unit: str = "",
):
    r"""Check inputs fall within bounds.

    This is a simple pass through function that tests whether the values fall within
    the bounds specified and issues a warning when this is not the case

    Args:
        values: An np.ndarray object or number
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

    # Implement the interval type
    if interval_type not in ["[]", "()", "[)", "(]"]:
        raise RuntimeWarning(f"Unknown interval type: {interval_type}")

    if interval_type[0] == "[":
        lower_func = np.less
    else:
        lower_func = np.less_equal

    if interval_type[1] == "]":
        upper_func = np.greater
    else:
        upper_func = np.greater_equal

    # Do the input values contain out of bound values? These tests are not
    # sensitive to dtype, float or int inputs and return either numpy.bool_
    # or np.ndarray with dtype 'bool', both of which support the sum() method
    out_of_bounds = np.logical_or(lower_func(values, lower), upper_func(values, upper))

    if out_of_bounds.sum():
        warnings.warn(
            f"Variable {label} ({unit}) contains values outside "
            f"the expected range ({lower},{upper}). Check units?"
        )

    return values


def input_mask(
    inputs: Union[np.ndarray, Number],
    lower: Number = -np.infty,
    upper: Number = np.infty,
    interval_type: str = "[]",
    label: str = "",
):
    r"""Mask inputs that do not fall within bounds.

    This function constrains the values in inputs, replacing values outside
    the provided interval with np.nan. Because np.nan is a float, when any data
    is out of bounds, the returned values are always float arrays or np.nan.

    Args:
        inputs: An np.ndarray object or number
        lower: The value of the lower constraint
        upper: The value of the upper constraint
        interval_type: The interval type of the constraint ('[]', '()', '[)', '(]')
        label: A string giving a descriptive label of the constrained contents
            used in reporting.

    Returns:
        If no data is out of bounds, the original inputs are returned, otherwise
        a float np.ndarray object with out of bounds values replaced with np.nan
        or np.nan for number or zero dimension ndarrays.

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

    # Implement the interval type
    if interval_type not in ["[]", "()", "[)", "(]"]:
        raise RuntimeWarning(f"Unknown interval type: {interval_type}")

    if interval_type[0] == "[":
        lower_func = np.less
    else:
        lower_func = np.less_equal

    if interval_type[1] == "]":
        upper_func = np.greater
    else:
        upper_func = np.greater_equal

    # Do the input values contain out of bound values? These tests are not
    # sensitive to dtype, float or int inputs and return either numpy.bool_
    # or np.ndarray with dtype 'bool', both of which support the sum() method
    mask = np.logical_or(lower_func(inputs, lower), upper_func(inputs, upper))

    # Check if any masking needs to be done
    if mask.sum():

        if isinstance(mask, np.bool_):
            # If mask is np.bool_ then a scalar or zero dimension ndarray was passed,
            # so return np.nan, implicitly converting the input to a float.

            warnings.warn(
                f"Scalar value {inputs} set to NaN "
                f"using {interval_type[0]}{lower}, {upper}{interval_type[1]} "
                f"bounds on {label}",
                category=RuntimeWarning,
            )

            return np.nan

        if isinstance(mask, np.ndarray):
            # If an ndarray, then we need a float version to set np.nan and we
            # copy to avoid modifying the original input
            if not np.issubdtype(inputs.dtype, np.floating):
                # Copies implicitly
                outputs = inputs.astype(np.float)
            else:
                outputs = inputs.copy()

            # Count the existing number of NaN values - impossible to have nan
            # in an integer input but isnan works with any input.
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

        raise NotImplementedError(f"Cannot set bounds on {type(inputs)}")

    return inputs
