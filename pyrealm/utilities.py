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
    finer set of datetimes, both of which need to be provided as np.datetime64
    arrays. Once an instance has been created, it can be called to interpolate a
    provided variable from the coarser to the finer timescale.

    Interpolation uses :func:`scipy.interpolate.interp1d`. This provides a range
    of interpolation kinds available, but this class adds a method
    `daily_constant`, which is basically the existing `previous` kind but offset
    so that a single value in a day is used for _all_ interpolated values in the
    day, including extrapolation backwards and forwards to midnight.

    Creating an instance sets up the interpolation time scales, and the instance
    can be called directly to interpolate a specific set of values.

    Args:
        input_datetimes: A numpy array giving the datetimes of the
            observations
        interpolation_datetimes: A numpy array giving the points at which to
            interpolate the observations.
    """

    def __init__(
        self,
        input_datetimes: np.ndarray,
        interpolation_datetimes: np.ndarray,
        method: str = "daily_constant",
    ) -> None:

        # There are some fussy things here with what is acceptable input to
        # interp1d: although the interpolation function can be created with
        # datetime.datetime or np.datetime64 values, the interpolation call
        # _must_ use float inputs (see https://github.com/scipy/scipy/issues/11093),
        # so this class uses floats from the inputs for interpolation.

        # Inputs must be datetime64 arrays and have the same temporal resolution
        # or the interpolation blows up.
        if not (
            np.issubdtype(input_datetimes.dtype, np.datetime64)
            and np.issubdtype(interpolation_datetimes.dtype, np.datetime64)
        ):
            raise ValueError("Interpolation times must be np.datetime64 arrays.")

        if input_datetimes.dtype != interpolation_datetimes.dtype:
            raise ValueError(
                "Interpolation times must have the same temporal resolution."
            )

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
            self._input_x = np.array(midnight).astype("float")

        else:
            self._input_x = input_datetimes.astype("float")

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
        # TODO - extend to 2 and 3d values, need to specify which axis is time.

        if self._method == "daily_constant":
            # Append the last value to match the appended day
            values = np.array(values)
            values = np.append(values, values[-1])
            method = "previous"
        else:
            method = self._method

        interp_fun = interp1d(self._input_x, values, kind=method, bounds_error=False)

        return interp_fun(self._interpolation_x)


class DailyRepresentativeValues:
    """Calculate daily representative values.

    This class is used to take data at a subdaily scale and extract a daily
    representative value. Creating an instance establishes the indices to be in
    extracting representative values and returns a callable instance that can be
    used to apply the calculation to different data arrays.

    The class implements three subsetting approaches:

    * a time window within each day with a given central time and width,
    * a window around the time of the maximum in a variable, and
    * using an existing boolean index, such as a predefined vector of night
        and day.

    The class provides the following public attributes:
    """

    def __init__(  # noqa C901
        self,
        datetimes: np.ndarray,
        window_center: float = None,
        window_width: float = None,
        include: np.ndarray = None,
        around_max: np.ndarray = None,
        reference_time: float = None,
    ) -> None:

        # TODO: - if the datetimes _are_ increasing and evenly spaced then
        # this could be coerced into a 2d array - which might speed up
        # the calculation of the reference_datetime_index?

        # Check the datetimes are strictly increasing and evenly spaced
        datetime_deltas = np.diff(datetimes)

        if not np.all(datetime_deltas == datetime_deltas[0]):
            raise ValueError("Datetime sequence must be evenly spaced")

        if datetime_deltas[0] < (datetimes[0] - datetimes[0]):
            raise ValueError("Datetime sequence must be increasing")

        # Get date sequence and unique dates (guaranteeing order of occurrence)
        self.datetime_shape = datetimes.shape
        self._date = datetimes.astype("datetime64[D]")
        _, idx = np.unique(self._date, return_index=True)
        self._date_sequence = self._date[np.sort(idx)]

        # Different methods
        if window_center is not None and window_width is not None:

            # Find which datetimes fall within that window, using second resolution
            win_center = np.timedelta64(int(window_center * 60 * 60), "s")
            win_start = np.timedelta64(
                int((window_center - window_width) * 60 * 60), "s"
            )
            win_end = np.timedelta64(int((window_center + window_width) * 60 * 60), "s")

            # Does that include more than one day?
            # NOTE - this might actually be needed at some point!
            if win_start < np.timedelta64(0, "s") or win_end > np.timedelta64(
                86400, "s"
            ):
                raise NotImplementedError(
                    "window_center and window_width cover more than one day"
                )

            # Now find which datetimes fall within that time window, given the
            # extracted dates
            self._include = np.logical_and(
                datetimes >= self._date + win_start, datetimes <= self._date + win_end
            )

            default_reference_datetime = self._date_sequence + win_center

        elif include is not None:

            if datetimes.shape != include.shape:
                raise RuntimeError("Datetimes and include do not have the same shape")

            if include.dtype != np.bool:
                raise RuntimeError("Include must be a boolean array.")

            self._include = include

            # Noon default reference time
            default_reference_datetime = self._date_sequence + np.timedelta64(
                43200, "s"
            )

        elif around_max is not None and window_width is not None:

            if datetimes.shape != around_max.shape:
                raise RuntimeError(
                    "Datetimes and around_max do not have the same shape"
                )

            raise NotImplementedError("around_max not yet implemented")

        else:

            raise RuntimeError("Unknown option combination")

        # Override the reference time from the default if provided
        if reference_time is not None:
            reference_time = np.timedelta64(int(reference_time * 60 * 60), "s")
            default_reference_datetime = self._date_sequence + reference_time

        # Store the reference_datetime
        self.reference_datetime = default_reference_datetime

        # Provide an index used to pull out daily reference values - it is possible
        # that the user might provide settings that don't match exactly to a datetime,
        # so use proximity.
        self.reference_datetime_idx = np.array(
            [
                np.argmin(np.abs(np.array(datetimes) - d))
                for d in self.reference_datetime
            ]
        )

    def __call__(
        self, values: np.ndarray, with_reference_values: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Extract the representative values for a variable.

        Calling an instance of
        :class:`~pyrealm.utilties.DailyRepresentativeValue` takes the provided
        variable and extracts the representative values for each day, given the
        configured methods in the instance.

        Some methods distinguish between a representative value, calculated over
        a time span, and a _specific_ reference value (e.g. noon). The
        `with_reference_value` argument species whether only the representative
        values should be return or if a 2-tuple of arrays containing the
        reference and representative values should be returned.

        Args:
            with_reference_values: A flag to request that representative values
                should also be returned.

        Returns:
            Either an np.ndarray of representative values or a 2-tuple of
            np.ndarrays containing the representative and reference values.
        """
        # https://vladfeinberg.com/2021/01/07/vectorizing-ragged-arrays.html

        if values.shape != self._date.shape:
            raise RuntimeError(
                "Values are not of the same shape as the datetime sequence"
            )

        average_values = np.empty_like(self._date_sequence, dtype=np.float)

        for idx, this_date in enumerate(self._date_sequence):
            # Get the mean of the values from this date that are included
            average_values[idx] = np.mean(
                values[np.logical_and(self._date == this_date, self._include)]
            )

        if with_reference_values:
            # Get the reference value and return that as well as daily value
            reference_values = values[self.reference_datetime_idx]
            return average_values, reference_values
        else:
            return average_values
