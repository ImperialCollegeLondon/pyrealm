"""Draft code for subdaily interpolators."""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore


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
