"""Draft code for subdaily interpolators."""  # noqa: D205, D415

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

    This class provides daily representative values from multiday datasets sampled at
    subdaily resolutions. An instance is created using a 1 dimensional
    :class:`numpy.datetime64` array providing the sampling times of observed data. These
    datetimes must be strictly increasing and the observations must be at the same time
    points during the day for a complete number of days.

    The cl


    aily subsets. Some use cases distinguish between a representative value, calculated
    over a time span, and a specific single value closest to a  _reference time_ (e.g.
    noon).

    The class provides three subsetting approaches:

    * A time window within each day, given the time of the window centre and its width.
      The default reference time is the window centre.
    * A boolean index of values to include, such as a predefined vector of night and
      day. The default reference time is noon.
    * A window around the time of the daily maximum in a variable. The default reference
      time is the daily maximum. This is not yet implemented

    An instance is created using a 1 dimensional numpy array of dtype numpy.datetime64,
    which must be strictly increasing and evenly spaced. Once the instance is created,
    it is callable and can be used to return representative values using the initial
    settings for different variables.

    Args:
        datetimes: A sequence of datetimes for observations at a subdaily scale
    """

    def __init__(
        self,
        datetimes: NDArray,
    ) -> None:
        # Datetime validation. The inputs must be:
        # - one dimensional datetime64
        if (datetimes.ndim > 1) or not np.issubdtype(datetimes.dtype, np.datetime64):
            raise ValueError(
                "Datetimes are not a 1 dimensional array with dtype datetime64"
            )

        # - with strictly increasing time deltas that are both evenly spaced and evenly
        #   divisible into a day
        n_datetimes = datetimes.shape[0]
        datetime_deltas = np.diff(datetimes)
        spacing = set(datetime_deltas)

        if len(spacing) > 1:
            raise ValueError("Datetime sequence not evenly spaced")

        self.spacing: np.timedelta64 = spacing.pop()
        """The time interval between observations"""

        if self.spacing < 0:
            raise ValueError("Datetime sequence must be increasing")

        # Get the number of observations per day and check it is evenly divisible.
        n_sec = 24 * 60 * 60
        obs_per_date = n_sec // self.spacing.astype("timedelta64[s]").astype(int)
        day_remainder = n_sec % self.spacing.astype("timedelta64[s]").astype(int)

        if day_remainder:
            raise ValueError("Datetime spacing is not evenly divisible into a day")

        obs_remainder = n_datetimes % obs_per_date
        if obs_remainder:
            raise ValueError("Datetimes include incomplete days")

        # Get a view of the datetimes wrapped on the number of observations per date
        # and extract the observation dates and times
        datetimes_by_date = datetimes.view()
        datetimes_by_date.shape = (-1, obs_per_date)

        # Data could still wrap onto obs x day view but having dates change mid row
        first_row_dates = datetimes_by_date[0, :].astype("datetime64[D]")
        if len(set(first_row_dates)) > 1:
            raise ValueError("Datetimes include incomplete days")

        self.observation_dates: NDArray = datetimes_by_date[:, 0].astype(
            "datetime64[D]"
        )
        """The dates covered by the observations"""

        self.n_days = len(self.observation_dates)
        """The number of days covered by the observations"""

        self.observation_times: NDArray = (
            datetimes_by_date[0, :] - self.observation_dates[0]
        ).astype("timedelta64[s]")
        """The times of observations through the day as timedelta64 values in seconds"""

        self.n_obs = len(self.observation_times)
        """The number of daily observations"""

        self.datetimes = datetimes
        """The datetimes used to create the instance."""

        self.include: NDArray[np.bool_]
        """A logical array indicating which values to be included in daily summaries.

        This attribute is only populated when one of the ``set_`` methods is called.
        """
        # Get the date change indices
        # date_change_idx = np.arange(0, self.n_datetimes + 1, obs_per_date)

        # self.dates = observation_dates[
        #     np.concatenate([date_change_idx, [self.n_datetimes - 1]])
        # ]

        # self.observation_dates = observation_dates
        # self.observation_times = observation_times.astype("timedelta64[s]")

    def set_window(self, window_center: float, window_width: float) -> None:
        """Set a daily window to sample.

        This method defines a time window within each day, given the time of the window
        centre and its width.

        Args:
            window_center: The centre of the time window in decimal hours
            window_width: The width of the time window in decimal hours
        """

        # TODO - use datetime64 and timedelta64 as inputs?

        # Find which datetimes fall within that window, using second resolution
        win_center = np.timedelta64(int(window_center * 60 * 60), "s")
        win_width = np.timedelta64(int(window_width * 60 * 60), "s")
        win_start = win_center - win_width
        win_end = win_center + win_width

        # Does that include more than one day?
        # NOTE - this might actually be needed at some point!
        if (win_start < 0) or (win_end > 86400):
            raise ValueError("window_center and window_width cover more than one day")

        # Now find which observation fall inclusively within that time window
        self.include = np.logical_and(
            win_start <= self.observation_times,
            win_end >= self.observation_times,
        )
        self.set_method = f"Window ({window_center}, {window_width})"

    def set_include(self, include: NDArray[np.bool_]) -> None:
        """TBD."""
        if self.datetimes.shape != include.shape:
            raise ValueError("Datetimes and include do not have the same shape")

        if include.dtype != bool:
            raise ValueError("The include argument must be a boolean array")

        # Noon default reference time
        self.include = include
        self.method = "Include array"

    def set_around_max(self, window_width: float) -> None:
        """TBD."""
        # This would have to be implemented _per_ value set, so in __call__
        # but can use date_change set up in init.
        raise NotImplementedError("around_max not yet implemented")

        self.method = "Around max"

        # # The approach implemented here uses cumsum and then divide by n_obs to
        # # quickly get mean values across ndarrays, even allowing for ragged
        # # arrays coming from the include option. The approach needs the indices
        # # of the values to include along the time axis (0), the index at which
        # # dates change those indices and the number of indices per group.
        # #
        # # See:
        # #    https://vladfeinberg.com/2021/01/07/vectorizing-ragged-arrays.html)

        # # Get a sequence of the indices of included values
        # self._include_idx = np.nonzero(include)[0]

        # # Find the last index for each date in that sequence, including the last
        # # value as the last index for the last date.
        # date_change = np.nonzero(np.diff(observation_dates[self._include_idx]))[0]
        # self._date_change = np.append(date_change, self._include_idx.shape[0] - 1)

        # # Count how many values included for each date
        # self._include_count = np.diff(self._date_change, prepend=-1)

        # # Override the reference time from the default if provided
        # if reference_time is not None:
        #     _reference_time = np.timedelta64(int(reference_time * 60 * 60), "s")
        #     default_reference_datetime = self.dates + _reference_time

        # # Store the reference_datetime
        # self._reference_datetime = default_reference_datetime

        # # Provide an index used to pull out daily reference values - it is possible
        # # that the user might provide settings that don't match exactly to a datetime,
        # # so use proximity.
        # self._reference_datetime_idx = np.array(
        #     [
        #         np.argmin(np.abs(np.array(datetimes) - d))
        #         for d in self._reference_datetime
        #     ]
        # )

    def get_representative_values(self, values: NDArray) -> NDArray:
        """Extract representative values for a variable.

        This method takes an array of values which has the same shape alo

        Args:
            values: An array of values for each observation, to be used to calculate
                representative daily values.

        Returns:
            An array of representative values
        """

        if not hasattr(self, "include"):
            raise AttributeError(
                "Use a set_ method to select which daily values are included"
            )
        # Check that the first axis has the same shape as the number of
        # datetimes in the init
        if values.shape[0] != self.datetimes.shape[0]:
            raise ValueError(
                "The first dimension of values is not the same length "
                "as the datetime sequence"
            )

        # Get a view of the values wrapped by date and then reshape along the first
        # axis into daily subarrays, leaving any remaining dimensions untouched.
        # Using a view and reshape should avoid copying the data.
        values_by_day = values.view()
        values_by_day.shape = tuple([self.n_days, self.n_obs] + list(values.shape[1:]))

        # subset to the included daily values
        return values_by_day[:, self.include, ...]

    def get_daily_means(self, values: NDArray) -> NDArray:
        """TBD."""
        daily_values = self.get_representative_values(values)

        return daily_values.mean(axis=1)
