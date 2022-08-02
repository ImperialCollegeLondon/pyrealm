"""The utilities module.

This module provides utility functions shared by modules and provides conversion
functions for common forcing variable inputs, such as hygrometric and radiation
conversions.
"""

from typing import List, Tuple, Union

import numpy as np
import tabulate
from scipy.interpolate import interp1d

from pyrealm.bounds_checker import bounds_checker
from pyrealm.param_classes import HygroParams

# from pandas.core.series import Series


def check_input_shapes(*args) -> Union[np.ndarray, int]:
    """Check sets of input variables have congruent shapes.

    This helper function validates inputs to check that they are either
    scalars or arrays and then that any arrays of the same shape. It either
    raises an error or returns the common shape or 1 if all arguments are
    scalar.

    Parameters:
        *args: A set of numpy arrays or scalar values

    Returns:
        The common shape of any array inputs or 1 if all inputs are scalar.

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

    return 1


def summarize_attrs(
    obj: object, attrs: List, dp: int = 2, repr_head: bool = True
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
                data = data.filled(np.nan)

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


# Psychrometric conversions to VPD for vapour pressure, specific humidity and
# relative humidity. Using the bigleaf R package as a checking reference from
# which the doctest values are taken


def calc_vp_sat(ta, hygro_params=HygroParams()):
    r"""Calculate vapour pressure of saturated air.

    This function calculates the vapour pressure of saturated air at a given
    temperature in kPa, using the Magnus equation:

    .. math::

        P = a \exp\(\frac{b - T}{T + c}\)

    The parameters :math:`a,b,c` can provided as a tuple, but three built-in
    options can be selected using a string.

    * ``Allen1998``: (610.8, 17.27, 237.3)
    * ``Alduchov1996``: (610.94, 17.625, 243.04)
    * ``Sonntag1990``: (611.2, 17.62, 243.12)

    Args:
        ta: The air temperature
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        Saturated air vapour pressure in kPa.

    Examples:
        >>> # Saturated vapour pressure at 21°C
        >>> round(calc_vp_sat(21), 6)
        2.480904
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(calc_vp_sat(21, hygro_params=allen), 6)
        2.487005
        >>> alduchov = HygroParams(magnus_option='Alduchov1996')
        >>> round(calc_vp_sat(21, hygro_params=alduchov), 6)
        2.481888
    """

    # Magnus equation and conversion to kPa
    cf = hygro_params.magnus_coef
    vp_sat = cf[0] * np.exp((cf[1] * ta) / (cf[2] + ta)) / 1000

    return vp_sat


def convert_vp_to_vpd(vp, ta, hygro_params=HygroParams()):
    """Convert vapour pressure to vapour pressure deficit.

    Args:
        vp: The vapour pressure in kPa
        ta: The air temperature in °C
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> round(convert_vp_to_vpd(1.9, 21), 7)
        0.5809042
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(convert_vp_to_vpd(1.9, 21, hygro_params=allen), 7)
        0.5870054
    """
    vp_sat = calc_vp_sat(ta, hygro_params=hygro_params)

    return vp_sat - vp


def convert_rh_to_vpd(rh, ta, hygro_params=HygroParams()):
    """Convert relative humidity to vapour pressure deficit.

    Args:
        rh: The relative humidity (proportion in (0,1))
        ta: The air temperature in °C
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> round(convert_rh_to_vpd(0.7, 21), 7)
        0.7442712
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(convert_rh_to_vpd(0.7, 21, hygro_params=allen), 7)
        0.7461016
        >>> import sys; sys.stderr = sys.stdout
        >>> round(convert_rh_to_vpd(70, 21), 7) #doctest: +ELLIPSIS
        pyrealm... contains values outside the expected range (0,1). Check units?
        -171.1823864
    """

    rh = bounds_checker(rh, 0, 1, "[]", "rh", "proportion")

    vp_sat = calc_vp_sat(ta, hygro_params=hygro_params)

    return vp_sat - (rh * vp_sat)


def convert_sh_to_vp(sh, patm, hygro_params=HygroParams()):
    """Convert specific humidity to vapour pressure.

    Args:
        sh: The specific humidity in kg kg-1
        patm: The atmospheric pressure in kPa
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure in kPa

    Examples:
        >>> round(convert_sh_to_vp(0.006, 99.024), 7)
        0.9517451
    """

    return sh * patm / ((1.0 - hygro_params.mwr) * sh + hygro_params.mwr)


def convert_sh_to_vpd(sh, ta, patm, hygro_params=HygroParams()):
    """Convert specific humidity to vapour pressure deficit.

    Args:
        sh: The specific humidity in kg kg-1
        ta: The air temperature in °C
        patm: The atmospheric pressure in kPa
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> round(convert_sh_to_vpd(0.006, 21, 99.024), 6)
        1.529159
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(convert_sh_to_vpd(0.006, 21, 99.024, hygro_params=allen), 5)
        1.53526
    """

    vp_sat = calc_vp_sat(ta, hygro_params=hygro_params)
    vp = convert_sh_to_vp(sh, patm, hygro_params=hygro_params)

    return vp_sat - vp


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
        input_datetimes: np.ndarray,
        interpolation_datetimes: np.ndarray,
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

    def __call__(self, values: np.ndarray):
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

    def __init__(  # noqa C901
        self,
        datetimes: np.ndarray,
        window_center: float = None,
        window_width: float = None,
        include: np.ndarray = None,
        around_max: bool = None,
        reference_time: float = None,
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
            reference_time = np.timedelta64(int(reference_time * 60 * 60), "s")
            default_reference_datetime = self.dates + reference_time

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
        self, values: np.ndarray, with_reference_values: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
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
            reference_values = values[self.reference_datetime_idx]
            return average_values, reference_values
        else:
            return average_values
