"""The :mod:`~pyrealm.utilities` module provides shared utility functions used to:

* check input arrays to functions have congruent shapes
* summarize object attributes in a tabular format
* apply bounds checking to inputs to functions and methods in the ``pyrealm`` package.

Some functions in ``pyrealm`` are only well-behaved with given bounds and bounds
checking also provides a (partial) check on the units being provided.

An earlier implementation of bounds checking used a subclass of
:class:`numpy.ma.MaskedArray` . The intention was that a checked array becomes a thing
that carries with it a description of any applied constraint, which sounds appealing
but...

1) MaskedArrays have a performance hit.
2) Subclasses are 'contagious': so x * subclass returns an object of class subclass.
   That then requires the subclass to be extended to handle all required methods.
3) NaN handling. Numpy has np.nan - and masked arrays do handle _masked_ np.nan values,
   but not unmasked ones:

.. :code-block: python

  >>> import numpy as np
  >>> x = np.ma.masked_array([1,2,np.nan, 4], mask=[1,0,0,1])
  >>> x.mean()
  nan
  >>> x = np.ma.masked_array([1,2,np.nan, 4], mask=[1,0,1,0])
  >>> x.mean()
  3.0

So the general approach here is now:

* a bounds checker function that tests if data exceeds bounds and warns when it does.
* a bounds mask function that returns np.ndarrays masked using np.nan when values are
  out of the applied bounds. Because np.nan is a _float_ value, masking is always
  applied to a float version of the input.

Note that this means that the codebase then needs to systematically use nan-aware
functions and possibly using bottleneck for speed. Part of the problem here is that
Numpy lacks a general solution to missing data:
https://numpy.org/neps/nep-0026-missing-data-summary
"""  # noqa: D205, D415

from typing import Union

import numpy as np
import tabulate
from numpy.typing import NDArray

from pyrealm import warnings


def check_input_shapes(*args: Union[float, int, np.generic, np.ndarray, None]) -> tuple:
    """Check sets of input variables have congruent shapes.

    This helper function validates inputs to check that they are either scalars or
    arrays and then that any arrays of the same shape. It returns a tuple of the common
    shape of the arguments, which is (1,) if all the arguments are scalar.

    Parameters:
        *args: A set of numpy arrays or scalar values

    Returns:
        The common shape of any array inputs or 1 if all inputs are scalar.

    Raises:
        ValueError: if the inputs contain arrays of differing shapes.

    Examples:
        >>> check_input_shapes(np.array([1,2,3]), 5)
        (3,)
        >>> check_input_shapes(4, 5)
        1
        >>> check_input_shapes(np.array([1,2,3]), np.array([1,2]))
        Traceback (most recent call last):
        ...
        ValueError: Inputs contain arrays of different shapes.
    """

    # Collect the shapes of the inputs
    shapes = set()

    # DESIGN NOTES - currently allow:
    #   - scalars,
    #   - 0 dim ndarrays (also scalars but packaged differently)
    #   - 1 dim ndarrays with only a single value

    for val in args:
        if isinstance(val, np.ndarray):
            # Note that 0-dim ndarrays (which are scalars) pass through as do
            # one dimensional arrays with a single value (also a scalar)
            if not (val.ndim == 0 or val.shape == (1,)):
                shapes.add(val.shape)
        # elif isinstance(val, Series):
        #    # Note that 0-dim ndarrays (which are scalars) pass through
        #    if val.ndim > 0:
        #        shapes.add(val.shape)
        elif val is None or isinstance(val, (float, int, np.generic)):
            pass  # No need to track scalars and optional values pass None
        else:
            raise ValueError(f"Unexpected input to check_input_shapes: {type(val)}")

    # shapes can be an empty set (all scalars) or contain one common shape
    # otherwise raise an error
    if len(shapes) > 1:
        raise ValueError("Inputs contain arrays of different shapes.")

    if len(shapes) == 1:
        return shapes.pop()

    return (1,)


def summarize_attrs(
    obj: object, attrs: list, dp: int = 2, repr_head: bool = True
) -> None:
    """Print a summary table of object attributes.

    This helper function prints a simple table of the mean, min, max and nan
    count for named attributes from an object for use in class summary
    functions.

    Args:
        obj: An object with attributes to summarize
        attrs: A list of strings of attribute names, or a list of 2-tuples
            giving attribute names and units.
        dp: The number of decimal places used in rounding summary stats.
        repr_head: A boolean indicating whether to show the object
            representation before the summary table.
    """

    # Check inputs
    if not isinstance(attrs, list):
        raise RuntimeError("attrs input not a list")

    # Create a list to hold variables and summary stats
    ret = []

    if len(attrs):
        first = attrs[0]

        # TODO: - not much checking for consistency here!
        if isinstance(first, str):
            has_units = False
            attrs = [(vl, None) for vl in attrs]
        else:
            has_units = True

        # Process the attributes
        for attr_entry in attrs:
            attr = attr_entry[0]
            unit = attr_entry[1]

            data = getattr(obj, attr)

            # Avoid masked arrays - run into problems with edge cases with all NaN
            if isinstance(data, np.ma.core.MaskedArray):
                # Mypy complains about filled being untyped?
                data = data.filled(np.nan)  # type: ignore

            # Add the variable and stats to the list to be displayed
            attr_row = [
                attr,
                np.round(np.nanmean(data), dp),
                np.round(np.nanmin(data), dp),
                np.round(np.nanmax(data), dp),
                np.count_nonzero(np.isnan(data)),
            ]
            if has_units:
                attr_row.append(unit)

            ret.append(attr_row)

    if has_units:
        hdrs = ["Attr", "Mean", "Min", "Max", "NaN", "Units"]
    else:
        hdrs = ["Attr", "Mean", "Min", "Max", "NaN"]

    if repr_head:
        print(obj)

    print(tabulate.tabulate(ret, headers=hdrs))


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
