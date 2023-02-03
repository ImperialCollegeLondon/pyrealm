"""The :mod:`~pyrealm.utilities` module provides shared utility functions used to:

* check input arrays to functions have congruent shapes
* summarize object attributes in a tabular format
* apply bounds checking to inputs to functions and methods in the :mod:`pyrealm`
  package.

Some functions in :mod:`pyrealm` are only well-behaved with given bounds and bounds
checking also provides a (partial) check on the units being provided.

An earlier implementation of bounds checking used a subclass of
:class:`numpy.ma.core.MaskedArray` .
The intention was that a checked array becomes a thing that carries with it a
description of any applied constraint, which sounds appealing but...

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
  - https://numpy.org/neps/nep-0026-missing-data-summary
"""  # noqa: D205, D415

from typing import Optional, Union

import numpy as np
import tabulate
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore

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
        A ValueError if the inputs contain arrays of differing shapes.

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


# New additions for subdaily Pmodel


class TemporalInterpolator:
    """Create a temporal interpolation of a variable.

    Instances of this class set a mapping from a coarser set of datetimes to a
    finer set of datetimes. Creating an instance sets up the interpolation time
    scales, and the instance can be called directly to interpolate a specific
    set of values.

    Interpolation uses :func:`scipy.interpolate.interp1d`. This provides a range
    of interpolation kinds available, with 'linear' probably the most
    appropriate, but this class adds a method `daily_constant`, which is
    basically the existing `previous` kind but offset so that a single value in
    a day is used for _all_ interpolated values in the day, including
    extrapolation backwards and forwards to midnight.

    Both inputs must be provided as arrays of type np.datetime64. This type has
    a range of subtypes with varying precision (e.g. 'datetime64[D]' for days
    and 'datetime64[s]' for seconds): the two input arrays _must_ use the same
    temporal resolution type.

    Args:
        input_datetimes: A numpy np.datetime64 array giving the datetimes of the
            observations
        interpolation_datetimes: A numpy np.datetime64 array giving the points
            at which to interpolate the observations.
    """

    def __init__(
        self,
        input_datetimes: NDArray[np.datetime64],
        interpolation_datetimes: NDArray[np.datetime64],
        method: str = "daily_constant",
    ) -> None:
        # This might be better as a straightforward function - there isn't a
        # huge amount of setup in __init__, so not saving a lot of processing by
        # saving that setup in class attributes for re-use.

        # TODO - this might be much more efficient with time as the _last_ axis
        # due to the organisation of contiguous memory in arrays.

        # There are some fussy things here with what is acceptable input to
        # interp1d: although the interpolation function can be created with
        # datetime.datetime or np.datetime64 values, the interpolation call
        # _must_ use float inputs (see https://github.com/scipy/scipy/issues/11093),
        # so internally this class uses floats from the inputs for
        # interpolation. Because the same datetime with different np.datetime64
        # subtypes gives different float values, the two arrays must use the
        # same subtype.

        # Inputs must be datetime64 arrays and have the same np.datetime64
        # subtype
        if not (
            np.issubdtype(input_datetimes.dtype, np.datetime64)
            and np.issubdtype(interpolation_datetimes.dtype, np.datetime64)
        ):
            raise TypeError("Interpolation times must be np.datetime64 arrays")

        if input_datetimes.dtype != interpolation_datetimes.dtype:
            raise TypeError("Inputs must use the same np.datetime64 precision subtype")

        self._method = method
        self._interpolation_x = interpolation_datetimes.astype("float")

        if method == "daily_constant":
            # This approach repeats the daily value for all subdaily times _on that
            # day_ and so should extrapolate beyond the last ref_datetime to the
            # end of that day and before the first ref_datetime to the beginning
            # of that day

            # Round observation times down to midnight on the day and append midnight
            # on the following day

            midnight = input_datetimes.astype("datetime64[D]").astype(
                input_datetimes.dtype
            )
            midnight = np.append(midnight, midnight[-1] + np.timedelta64(1, "D"))
            self._input_x = np.array(midnight).astype(float)

        else:
            self._input_x = input_datetimes.astype(float)

    def __call__(self, values: NDArray) -> NDArray:
        """Apply temporal interpolation to a variable.

        Calling an instance of :class:`~pyrealm.utilties.TemporalInterpolator`
        with a variable applies the temporal interpolation set in the instance
        to the inputs and returns an interpolated variable, using the method set
        when the instance was created.

        Args:
            values: A numpy array of numeric values, of the same length as the
            `input_datetimes` used to create the instance.

        Returns:
            A numpy array of values interpolated to the timepoints in the
            `interpolation_datetimes` values used to create the instance.
        """

        if self._method == "daily_constant":
            # Append the last value to match the appended day
            values = np.array(values)
            values = np.append(values, values[-1])
            method = "previous"
        else:
            method = self._method

        # Check the first axis of the values has the same length as the input
        # datetimes.
        if len(self._input_x) != values.shape[0]:
            raise ValueError(
                "The first axis of values does not match the length of input_datetimes"
            )

        interp_fun = interp1d(
            self._input_x, values, axis=0, kind=method, bounds_error=False
        )

        return interp_fun(self._interpolation_x)


class DailyRepresentativeValues:
    """Calculate daily representative values.

    This class is used to take data at a subdaily scale and calculate daily
    representative mean values across daily subsets. Some use cases
    distinguish between a representative value, calculated over a time span, and
    a specific single value closest to a  _reference time_ (e.g. noon).

    The class provides three subsetting approaches:

    * A time window within each day, given the time of the window centre and its
      width. The default reference time is the window centre.
    * A boolean index of values to include, such as a predefined vector of night
      and day. The default reference time is noon.
    * A window around the time of the daily maximum in a variable. The default
      reference time is the daily maximum. This is not yet implemented

    An instance is created using a 1 dimensional numpy array of dtype
    numpy.datetime64, which must be strictly increasing and evenly spaced. Once
    the instance is created, it is callable and can be used to return
    representative values using the initial settings for different variables.

    Args:
        datetimes: A sequence of datetimes for observations at a subdaily scale
        window_center: The centre of the time window in hours
        window_width: The width of the time window in hours
        include: A boolean vector showing indicating which observed values to
            include in calculating representative values
        around_max: A boolean flag to use representative values around the daily
            maximum
        reference_time: A time to be used for reference values in decimal hours,
            overriding the default time for the given method.

    Attributes:
        dates np.ndarray: The dates for calculated representative values.
        n_datetimes int: The number of observed datetimes
        method str: Summary of the method being used
    """

    def __init__(  # noqa C901 - Function is too complex
        self,
        datetimes: NDArray,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        include: Optional[np.ndarray] = None,
        around_max: Optional[bool] = None,
        reference_time: Optional[float] = None,
    ) -> None:
        # Datetime validation. The inputs must be:
        # - one dimensional datetime64
        # - with strictly increasing and evenly spaced time deltas
        # - covering a set of whole days
        if not (
            (len(datetimes.shape) == 1) & np.issubdtype(datetimes.dtype, np.datetime64)
        ):
            raise ValueError(
                "Datetimes are not a 1 dimensional array with dtype datetime64"
            )

        self.n_datetimes = datetimes.shape[0]
        datetime_deltas = np.diff(datetimes)

        if not np.all(datetime_deltas == datetime_deltas[0]):
            raise ValueError("Datetime sequence must be evenly spaced")

        if datetime_deltas[0] < 0:
            raise ValueError("Datetime sequence must be increasing")

        # The sequence is now strictly increasing and evenly spaced, so get the
        # indices of date changes to check whole days and get the date sequence
        observation_dates = datetimes.astype("datetime64[D]")
        date_change_idx = np.where(np.diff(observation_dates).astype(int) == 1)[0]

        # Get the count of observations per date - including last date change to
        # end of sequence
        obs_per_date = np.diff(
            np.concatenate([date_change_idx, [self.n_datetimes - 1]])
        )

        if not np.all(obs_per_date == obs_per_date[0]):
            raise ValueError("Datetime sequence does not cover a whole number of days")

        self.dates = observation_dates[
            np.concatenate([date_change_idx, [self.n_datetimes - 1]])
        ]

        # Different methods
        if window_center is not None and window_width is not None:
            # Find which datetimes fall within that window, using second resolution
            win_center = np.timedelta64(int(window_center * 60 * 60), "s")
            win_start = np.timedelta64(
                int((window_center - window_width / 2) * 60 * 60), "s"
            )
            win_end = np.timedelta64(
                int((window_center + window_width / 2) * 60 * 60), "s"
            )

            # Does that include more than one day?
            # NOTE - this might actually be needed at some point!
            if (win_start < np.timedelta64(0, "s")) or (
                win_end > np.timedelta64(86400, "s")
            ):
                raise NotImplementedError(
                    "window_center and window_width cover more than one day"
                )

            # Now find which datetimes fall within that time window, given the
            # extracted dates
            include = np.logical_and(
                datetimes >= observation_dates + win_start,
                datetimes <= observation_dates + win_end,
            )

            default_reference_datetime = self.dates + win_center

            self.method = f"Window ({window_center}, {window_width})"

        elif include is not None:
            if datetimes.shape != include.shape:
                raise ValueError("Datetimes and include do not have the same shape")

            if include.dtype != bool:
                raise ValueError("The include argument must be a boolean array")

            # Noon default reference time
            default_reference_datetime = self.dates + np.timedelta64(12, "h")

            self.method = "Include array"

        elif around_max is not None and window_width is not None:
            # This would have to be implemented _per_ value set, so in __call__
            # but can use date_change set up in init.
            raise NotImplementedError("around_max not yet implemented")

            self.method = "Around max"

        else:
            raise RuntimeError("Unknown option combination")

        # The approach implemented here uses cumsum and then divide by n_obs to
        # quickly get mean values across ndarrays, even allowing for ragged
        # arrays coming from the include option. The approach needs the indices
        # of the values to include along the time axis (0), the index at which
        # dates change those indices and the number of indices per group.
        #
        # See:
        #    https://vladfeinberg.com/2021/01/07/vectorizing-ragged-arrays.html)

        # Get a sequence of the indices of included values
        self._include_idx = np.nonzero(include)[0]

        # Find the last index for each date in that sequence, including the last
        # value as the last index for the last date.
        date_change = np.nonzero(np.diff(observation_dates[self._include_idx]))[0]
        self._date_change = np.append(date_change, self._include_idx.shape[0] - 1)

        # Count how many values included for each date
        self._include_count = np.diff(self._date_change, prepend=-1)

        # Override the reference time from the default if provided
        if reference_time is not None:
            _reference_time = np.timedelta64(int(reference_time * 60 * 60), "s")
            default_reference_datetime = self.dates + _reference_time

        # Store the reference_datetime
        self._reference_datetime = default_reference_datetime

        # Provide an index used to pull out daily reference values - it is possible
        # that the user might provide settings that don't match exactly to a datetime,
        # so use proximity.
        self._reference_datetime_idx = np.array(
            [
                np.argmin(np.abs(np.array(datetimes) - d))
                for d in self._reference_datetime
            ]
        )

    def __call__(
        self, values: NDArray, with_reference_values: bool = False
    ) -> Union[tuple[NDArray, NDArray], NDArray]:
        """Calculate representative values for a variable.

        Instances of :class:`~pyrealm.utilities.DailyRepresentativeValues` can
        be called to calcualte representative values for each day for a provided
        array of values, given the methods configured in the instance.

        Args:
            with_reference_values: A flag to request that reference values
                should be returned as well as the representative values.

        Returns:
            Either an np.ndarray of representative values or a 2-tuple of
            np.ndarrays containing the representative and reference values.
        """

        # Check that the first axis has the same shape as the number of
        # datetimes in the init
        if values.shape[0] != self.n_datetimes:
            raise ValueError(
                "The first dimension of values is not the same length "
                "as the datetime sequence"
            )

        # Get the cumulative sum of the included values, then reduce to the
        # sums at the indices where dates change
        values_cumsum = np.cumsum(values[self._include_idx], axis=0)[
            self._date_change, ...
        ]

        # Now find the differences across days to reduce to the sum of values
        # _within_ days and divide by the count to get averages. Need to take
        # care here to ensure that the counts always align along the first axis.
        count_shape = np.concatenate(
            [self._include_count.shape, np.repeat([1], values.ndim - 1)]
        )

        average_values = np.diff(
            values_cumsum, prepend=0, axis=0
        ) / self._include_count.reshape(count_shape)

        if with_reference_values:
            # Get the reference value and return that as well as daily value
            reference_values = values[self._reference_datetime_idx]
            return average_values, reference_values
        else:
            return average_values
