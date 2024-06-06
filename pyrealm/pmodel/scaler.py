"""The :mod:`~pyrealm.pmodel.scaler` module provides the
:class:`~pyrealm.pmodel.scaler.SubdailyScaler` class, which is a core
component of fitting the P Model at subdaily time scales. The class is used as follows:

* A :class:`~pyrealm.pmodel.scaler.SubdailyScaler` instance is created using
  the time series of the observations for the subdaily data being used within a model.

* An acclimation window is then set, defining a period of the day representing the
  environmental conditions that plants will acclimate to. This will typically by the
  time of day with highest productivity - usually around noon - when the light use
  efficiency of the plant can best make use of high levels of sunlight. The window can
  be set using one of three methods:

  * The :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.set_window` method sets a
    window centred on a given time during the day with a fixed width.
  * The :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.set_nearest` method sets
    the acclimation window as the single observation closest to a given time of day.
  * The :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.set_include` method
    allows the user to set an arbitrary selection of observations during the day as the
    acclimation window.

  If new ``set_`` functions are defined, then they will need to call the
  :meth:`~pyrealm.pmodel.scaler.SubdailyScaler._set_times` method to update
  the instance attributes used to set the acclimation window.

* The :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.get_daily_means` method
  can then be used to get the average value of a variable within the acclimation window
  for each day. Alternatively, the
  :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.get_window_values` method can
  be used to get the actual values observed during each daily window.

* The :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.fill_daily_to_subdaily`
  reverses this process: it takes an array of daily values and fills those values back
  onto the faster timescale used to create the
  :class:`~pyrealm.pmodel.scaler.SubdailyScaler` instance.
"""  # noqa: D205, D415

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore


class SubdailyScaler:
    """Convert variables between photosynthetic fast and slow response scales.

    This class provides methods that allow data to be converted between
    photsynthetically 'fast' and 'slow' timescales. Data on fast timescales capture
    subdaily variation in environmental conditions and slow timescales capture the
    timescales over which plants will acclimate to changing conditions.

    An instance of this class is created using an array of :class:`numpy.datetime64`
    values that provide the observation datetimes for a dataset sampled on a 'fast'
    timescale. The datetimes must:

    * be strictly increasing,
    * be evenly spaced, using a spacing that evenly divides a day, and
    * completely cover a set of days.

    Warning:
        The values in ``datetimes`` are assumed to be the precise times of the
        observations and are converted to second precision for internal calculations. If
        the datetimes are at a _coarser_ precision and represent a sampling time span,
        then they should first be converted to a reasonable choice of observation time,
        such as the midpoint of the timespan.

    Args:
        datetimes: A sequence of datetimes for observations at a subdaily scale.
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

        # - stored with second precision
        datetimes = datetimes.astype("datetime64[s]")

        # - with strictly increasing time deltas that are both evenly spaced and evenly
        #   divisible into a day
        datetime_deltas = np.diff(datetimes)
        spacing = set(datetime_deltas)

        if len(spacing) > 1:
            raise ValueError("Datetime sequence not evenly spaced")

        self.spacing: np.timedelta64 = spacing.pop()
        """The time interval between observations"""

        if self.spacing < 0:
            raise ValueError("Datetime sequence must be increasing")

        # Work out observations by date and look for incomplete dates at the start and
        # end of the time sequence
        dates = datetimes.astype("datetime64[D]")
        unique_dates, date_counts = np.unique(dates, return_counts=True)

        # Do we have at least three complete day to ensure we have a complete set of
        # observation times on day 2. Three days is not sensible for the model but this
        # ensures the padding can de determined.
        if len(unique_dates) < 3:
            ValueError("Not enough data to validate observation times")

        # Get the maximum number of observations per day and check it is evenly
        # divisible by the number of seconds in a day.
        obs_per_date = date_counts.max()
        day_remainder = (24 * 60 * 60) % obs_per_date
        if day_remainder:
            raise ValueError("Datetime spacing is not evenly divisible into a day")

        # Set the datetime padding
        self.padding: tuple[int, int] = (
            obs_per_date - date_counts[0],
            obs_per_date - date_counts[-1],
        )
        """Padding used to handle incomplete start and end days.

        Provides the number of observations to add to the start and end of the provided
        datetime sequence to give complete days."""

        # Get a complete set of observation times from day 2 - we have guaranteed
        # that the second day is complete.
        complete_times = datetimes[dates == unique_dates[1]]
        observation_timedeltas = complete_times - complete_times[0].astype(
            "datetime64[D]"
        )

        # Add padded datetimes as a time series containing only complete days, using an
        # offset to the last observations to avoid an off by one error from np.arange.
        self.padded_datetimes = np.arange(
            unique_dates[0] + observation_timedeltas[0],
            unique_dates[-1] + (observation_timedeltas[-1] + np.timedelta64(1, "s")),
            np.diff(observation_timedeltas)[0],
        )

        self.observation_dates: NDArray[np.datetime64] = unique_dates
        """The dates covered by the observations"""

        self.n_days: int = len(self.observation_dates)
        """The number of days covered by the observations"""

        self.observation_times: NDArray[np.timedelta64] = observation_timedeltas
        """The times of observations through the day as timedelta64 values in seconds"""

        self.n_obs: int = len(self.observation_times)
        """The number of daily observations"""

        self.datetimes: NDArray[np.datetime64] = datetimes
        """The datetimes used to create the instance."""

        # The following attributes are only set when one of the set methods is called.

        self.include: NDArray[np.bool_]
        """Logical index of which daily observations are included in daily samples.
        """

        self.sample_datetimes: NDArray[np.datetime64]
        """Datetimes included in daily samples.

        This array is two-dimensional: the first axis is of length n_days and the second
        axis is of length n_samples.
        """

        self.sample_datetimes_mean: NDArray[np.datetime64]
        """The mean datetime of for each daily sample."""

        self.sample_datetimes_max: NDArray[np.datetime64]
        """The maximum datetime of for each daily sample."""

    def _set_times(self) -> None:
        """Sets the times at which representative values are sampled.

        This private method should be called by all ``set_`` methods. It is used to
        update the instance to populate the following attributes:

        * :attr:`~pyrealm.pmodel.scaler.SubdailyScaler.sample_datetimes`: An
          array of the datetimes of observations included in daily samples of shape
          (n_day, n_sample).
        * :attr:`~pyrealm.pmodel.scaler.SubdailyScaler.sample_datetimes_mean`:
          An array of the mean daily datetime of observations included in daily samples.
        * :attr:`~pyrealm.pmodel.scaler.SubdailyScaler.sample_datetimes_max`:
          An array of the maximum daily datetime of observations included in daily
          samples.
        """

        # Get a view of the complete padded datetimes and then reshape along the first
        # axis into daily subarrays.
        times_by_day = self.padded_datetimes.view()
        times_by_day.shape = tuple([self.n_days, self.n_obs])

        # subset to the included daily values
        self.sample_datetimes = times_by_day[:, self.include]

        # Cannot use mean directly, so calculate the mean of the values expressed
        # as seconds and then add the mean values back onto the epoch.
        now = np.datetime64("now", "s")
        epoch = now - np.timedelta64(now.astype(int), "s")
        datetimes_as_seconds = self.sample_datetimes.astype(int)
        mean_since_epoch = datetimes_as_seconds.mean(axis=1).round().astype(int)
        max_since_epoch = datetimes_as_seconds.max(axis=1).round().astype(int)

        self.sample_datetimes_mean = epoch + mean_since_epoch.astype("timedelta64[s]")
        self.sample_datetimes_max = epoch + max_since_epoch.astype("timedelta64[s]")

    def set_window(
        self, window_center: np.timedelta64, half_width: np.timedelta64
    ) -> None:
        """Set a daily window to sample.

        This method sets the daily values to sample using a time window, given the time
        of the window centre and its width. Both of these values must be provided as
        :class:`~numpy.timedelta64` values.

        Args:
            window_center: A timedelta since midnight to use as the window center
            half_width: A timedelta to use as a window width on each side of the center
        """

        if not (
            isinstance(window_center, np.timedelta64)
            and isinstance(half_width, np.timedelta64)
        ):
            raise ValueError(
                "window_center and half_width must be np.timedelta64 values"
            )

        # Find the timedeltas of the window start and end, using second resolution
        win_start = (window_center - half_width).astype("timedelta64[s]")
        win_end = (window_center + half_width).astype("timedelta64[s]")

        # Does that include more than one day?
        # NOTE - this might actually be needed at some point!
        if (win_start < 0) or (win_end > 86400):
            raise ValueError("window_center and half_width cover more than one day")

        # Now find which observation fall inclusively within that time window
        self.include = np.logical_and(
            win_start <= self.observation_times,
            win_end >= self.observation_times,
        )
        self.set_method = f"Window ({window_center}, {half_width})"
        self._set_times()

    def set_include(self, include: NDArray[np.bool_]) -> None:
        """Set a sequence of daily values to sample.

        This method sets which daily values will be sampled directly, by providing a
        boolean array for the daily observation times. The ``include`` array must be of
        the same length as the number of daily observations.

        Args:
            include: A boolean array indicating which daily observations to include.
        """

        if not (isinstance(include, np.ndarray) and include.dtype == np.bool_):
            raise ValueError("The include argument must be a boolean array")

        if self.n_obs != len(include):
            raise ValueError("The include array length is of the wrong length")

        self.include = include
        self.method = "Include array"
        self._set_times()

    def set_nearest(self, time: np.timedelta64) -> None:
        """Sets a single observation closest to a target time to be sampled.

        This method finds the daily observation time closest to a value provided as a
        :class:`~numpy.timedelta64` value since midnight. If the provided time is
        exactly between two observation times, the earlier observation will be used.
        The resulting single observation will then as the daily sample.

        Args:
            time: A :class:`~numpy.timedelta64` value.
        """

        if not isinstance(time, np.timedelta64):
            raise ValueError("The time argument must be a timedelta64 value.")

        # Convert to seconds and check it is in range
        time = time.astype("timedelta64[s]")
        if not (time >= 0 and time < 24 * 60 * 60):
            raise ValueError("The time argument is not >= 0 and < 24 hours.")

        # Calculate the observation time closest to the provided value
        nearest = np.argmin(abs(self.observation_times - time))
        include = np.zeros(self.n_obs, dtype=np.bool_)
        include[nearest] = True

        self.include = include
        self.method = f"Set nearest: {time}"
        self._set_times()

    # def set_around_max(self, window_width: float) -> None:
    #     """

    #     """
    #     # This would have to be implemented _per_ value set, so in __call__
    #     # but can use date_change set up in init.
    #     raise NotImplementedError("around_max not yet implemented")

    #     self.method = "Around max"

    def pad_values(self, values: NDArray) -> NDArray:
        """Pad values array to full days.

        This method takes an array representing daily values and pads the first and
        last day with NaN values so that they correspond to full days, similar to the
        datetimes array of this class.

        Args:
            values: An array containing the sample values. The first dimension should be
              matching the number of measurements, i.e., the first dimension of
              datetimes in
              :class:`~pyrealm.pmodel.scaler.SubdailyScaler`.
        """

        if self.padding == (0, 0):
            return values

        # Construct a padding iterable for np.pad, to pad only the first time axis
        padding_dims = list((self.padding,))
        padding_dims.extend([(0, 0)] * (values.ndim - 1))

        return np.pad(values, padding_dims, constant_values=(np.nan, np.nan))

    def get_window_values(self, values: NDArray) -> NDArray:
        """Extract acclimation window values for a variable.

        This method takes an array of values which has the same shape along the first
        axis as the datetimes used to create the instance and extracts the values from
        the acclimation window set using one of the ``set_`` methods.

        Args:
            values: An array of values for each observation.

        Returns:
            An array of the values within the defined acclimation window
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
        # When the values are padded the returned value is automatically a padded copy
        # of the original data, but if there is no padding, the original data is
        # returned and using a view and reshape should avoid copying the data.
        padded_values = self.pad_values(values)
        values_by_day = padded_values.view()
        values_by_day.shape = tuple([self.n_days, self.n_obs, *list(values.shape[1:])])

        # subset to the included daily values
        return values_by_day[:, self.include, ...]

    def get_daily_means(
        self, values: NDArray, allow_partial_data: bool = False
    ) -> NDArray:
        """Get the daily means of a variable during the acclimation window.

        This method extracts values from a given variable during a defined acclimation
        window set using one of the ``set_`` methods, and then calculates the daily mean
        of those values.

        The `allow_partial_data` option switches between using :func:`numpy.mean` and
        :func:`numpy.nanmean`, so that daily mean values can be calculated even if the
        data in the acclimation window is incomplete. Note that this will still return
        `np.nan` if _no_ data is present in the acclimation window. It also has no
        effect if the
        :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.set_nearest` method has
        been used to set the acclimation observations, because this method only ever
        sets a single observation.

        The values can have any number of dimensions, but the first dimension must
        represent the time axis and have the same length as the original set of
        observation times.

        Args:
            values: An array of values to reduce to daily averages.
            allow_partial_data: Exclude missing data from the calculation of the daily
                average value.

        Returns:
            An array of mean daily values during the acclimation window
        """
        daily_values = self.get_window_values(values)

        if allow_partial_data:
            return np.nanmean(daily_values, axis=1)

        return daily_values.mean(axis=1)

    def fill_daily_to_subdaily(
        self,
        values: NDArray,
        update_point: str = "max",
        kind: str = "previous",
        fill_from: np.timedelta64 | None = None,
    ) -> NDArray:
        """Resample daily variables onto the subdaily time scale.

        This method takes an array representing daily values and interpolates those
        values back onto the subdaily timescale used to create the
        :class:`~pyrealm.pmodel.scaler.SubdailyScaler` instance. The first
        axis of the `values` must be the same length as the number of days used to
        create the instance.

        Two interpolation kinds are currently implemented:

        * ``previous`` interpolates the daily value as a constant, until updating to the
          next daily value. This option will fill values until the end of the time
          series.
        * ``linear`` interpolates linearly between the update points of the daily
          values. The interpolated values are held constant for the first day and then
          interpolated linearly: this is to avoid plants adapting optimally to future
          conditions.

        The update point defaults to the maximum time of day during the acclimation
        window. It can also be set to the mean time of day, but note that this implies
        that the plant predicts the daily values between the mean and max observation
        time. The ``fill_from`` argument can be used to set the update point to an
        arbitrary time of day.

        Args:
            values: An array with the first dimension matching the number of days in the
              instances :class:`~pyrealm.pmodel.scaler.SubdailyScaler` object.
            update_point: The point in the acclimation window at which the plant updates
              to the new daily value: one of 'mean' or 'max'.
            kind: The kind of interpolation to use to fill between daily values: one of
              'previous' or 'linear',
            fill_from: As an alternative to ``update_point``, an
              :class:`numpy.timedelta64` value giving the time of day from which to fill
              values forward.
        """

        if values.shape[0] != self.n_days:
            raise ValueError("Values is not of length n_days on its first axis.")

        if fill_from is not None:
            if not isinstance(fill_from, np.timedelta64):
                raise ValueError("The fill_from argument must be a timedelta64 value.")

            # Convert to seconds and check it is in range
            _fill_from = fill_from.astype("timedelta64[s]")
            if not (_fill_from >= 0 and _fill_from < 24 * 60 * 60):
                raise ValueError("The fill_from argument is not >= 0 and < 24 hours.")

            update_time = self.observation_dates + _fill_from

        elif update_point == "max":
            update_time = self.sample_datetimes_max
        elif update_point == "mean":
            update_time = self.sample_datetimes_mean
        else:
            raise ValueError("Unknown update point")

        # Note that interp1d cannot handle datetime64 inputs, so need to interpolate
        # using datetimes cast to integer types

        # Use fill_value to handle extrapolation before or after update point:
        # - previous will fill the last value forward to the end of the time series,
        #   although this will be bogus if the interpolation time series extends beyond
        #   the next update point
        # - linear has a 1 day offset: the plant cannot adapt towards the next optimal
        #   value until _after_ the update point.

        if kind == "previous":
            fill_value = (None, values[-1])
        elif kind == "linear":
            values = np.insert(values, 0, values[0], axis=0)
            update_time = np.append(
                update_time, update_time[-1] + np.timedelta64(1, "D")
            )
            fill_value = (None, None)
        else:
            raise ValueError("Unsupported interpolation option")

        interp_fun = interp1d(
            update_time.astype("int"),
            values,
            axis=0,
            kind=kind,
            bounds_error=False,
            fill_value=fill_value,
        )

        # TODO - The kind "previous" might be replaceable with bottleneck.push
        #
        # v = np.empty_like(tk)
        # v[:] = np.nan

        # v[values_idx] = values
        # v = bn.push(v)

        return interp_fun(self.datetimes.astype("int"))
