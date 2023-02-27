"""The :mod:`~pyrealm.subdaily.daily_values` module provides functionality to calculate
daily representative values from multiday datasets sampled at subdaily resolutions. This
is a crucial step in including photosynthetic acclimation and plant fast responses into
the P Model, because a plant will likely acclimate to conditions at a particular time of
day, such as around noon.

The :class:`~pyrealm.subdaily.daily_values.DailyRepresentativeValues` class provides
this functionality using the following workflow:

* A :class:`~pyrealm.subdaily.daily_values.DailyRepresentativeValues` instance is
  created using the observation times for the subdaily data being used within a model.
* One of a number of ``set_`` methods is used to set which observations within each day
  should be used to calculate daily values.
* The
  `:meth:`~pyrealm.subdaily.daily_values.DailyRepresentativeValues.get_representative_values`
  or  `:meth:`~pyrealm.subdaily.daily_values.DailyRepresentativeValues.get_daily_means`
  methods can then be used to extract the values for a particular variable.

  The variable passed to either of these methods can have any number of dimensions, but
  the first dimension must represent the time axis and must have the same length as the
  original set of observation times.

It is possible to use the different `set_` methods to change which values are being
extracted.
"""  # noqa: D205, D415

import numpy as np
from numpy.typing import NDArray


class DailyRepresentativeValues:
    """Extracting daily representative values.

    An instance of this class is created using an array of :class:`numpy.datetime64`
    values that provide the observation datetimes of data to be sample. These values
    must:

    * be strictly increasing,
    * be evenly spaced, using a spacing that evenly divides a day, and
    * completely cover a set of days.

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

    def set_window(self, window_center: float, window_width: float) -> None:
        """Set a daily window to sample.

        This method defines a time window within each day, given the time of the window
        centre and its width, both in decimal hours.

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

    def set_nearest(self, time: float) -> None:
        """Sets a single observation closest to a target time to be sampled.

        This method finds the daily observation time closest to a value provided in
        decimal hours. If the provided time is exactly between two observation times,
        the earlier observation will be used. The resulting single observation will then
        be used to extract representative values.

        Args:
            time: A float value in decimal hours.
        """

        if not (isinstance(time, float) and time >= 0 and time < 24):
            raise ValueError("The time argument must float in (0, 24].")

        # Calculate the observation time closest to the provided value
        time_td64 = np.timedelta64(int(time * 60 * 60), "s")
        nearest = np.argmin(abs(self.observation_times - time_td64))
        include = np.zeros(self.n_obs, dtype=np.bool_)
        include[nearest] = True

        self.include = include
        self.method = f"Set nearest: {time}"

    # def set_around_max(self, window_width: float) -> None:
    #     """

    #     """
    #     # This would have to be implemented _per_ value set, so in __call__
    #     # but can use date_change set up in init.
    #     raise NotImplementedError("around_max not yet implemented")

    #     self.method = "Around max"

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
