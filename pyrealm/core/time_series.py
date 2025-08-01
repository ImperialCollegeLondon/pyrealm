"""This module provides general tools for working with time series data. It currently
provides the :class:`AnnualValueCalculator`, which is used to calculate annual means and
totals of time series data, and :func:`broadcast_time`, which is used to broadcast
arrays over the time axis.
"""  # noqa : D205

from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from pyrealm.pmodel import AcclimationModel


class AnnualValueCalculator:
    """A calculator class for annual means and totals from time series data.

    This class is used to calculate annual means and totals from time series data. An
    instance is created by providing a set of timings for the times series data, either
    as a one-dimensional array of datetimes or as an AcclimationModel instance from a
    SubdailyPModel, which provides validated datetimes at subdaily temporal resolutions.

    The calculation process accounts for observations that span year boundaries, such as
    fortnightly data, by calculating the duration of each observation within each year.
    The process also handles unequal sampling intervals - such as monthly data - by
    calculating the actual duration of observations. However, with uneven sampling, the
    duration of the last interval is unknown and so an explicit endpoint must be
    provided.

    The indexing of annual subsets of observations, along with the appropriate
    weightings for observations values in calculating annual values, are calculated when
    the class is created and then used by the ``get_annual_means`` and
    ``get_annual_totals`` methods. Both methods return values for all years sampled:
    the ``year_completeness`` attribute records what fraction of a year has been sampled
    to give a particular value.

    .. Note::

        The class handles a wide range of different possible sampling frequencies and
        calculates weights for observations using the duration of observations with
        second precision. With uneven durations - such as monthly data - this will give
        slightly different values to simple means assuming equal duration.

    Example:
        >>> # Three years of monthly data
        >>> datetimes = np.arange(
        ...     np.datetime64('2000-01'),
        ...     np.datetime64('2003-01'),
        ...     np.timedelta64(1, "M")
        ... )
        >>> # Monthly data is uneven - requires an explicit endpoint.
        >>> avc = AnnualValueCalculator(datetimes, endpoint=np.datetime64('2003-01'))
        >>> avc.year_completeness
        array([1., 1., 1.])
    """

    def __init__(
        self,
        timing: AcclimationModel | NDArray[np.datetime64],
        growing_season: NDArray[np.bool_] | None = None,
        endpoint: np.datetime64 | None = None,
    ):
        # Attribute definitions
        self.datetimes: NDArray[np.datetime64]
        """The start datetime of observations taking from the initial timings"""
        self.n_obs: int
        """The number of observations in the time series."""
        self.endpoint: np.datetime64
        """A datetime giving of the end of the last observation."""
        self.growing_season: NDArray[np.bool_]
        """The initial input array of growing season data."""
        self.duration_seconds: NDArray[np.int_]
        """The duration of each observation in seconds."""
        self.indexing: list[tuple[int, int]] = []
        """Pairs of integers giving start and end indices to extract consecutive years
        of data from the time series."""
        self.duration_weights: list[NDArray[np.int_]] = []
        """A list of arrays giving the number of seconds that each observation
        within a year contributes to that year."""
        self.fractional_weights: list[NDArray[np.float64]] = []
        """A list of arrays giving the fraction of each observation within a year that
        falls in the year."""
        self.growing_season_by_year: list[NDArray[np.bool_]]
        """A list of arrays giving the growing season subarrays for each year."""
        self.year_completeness: NDArray[np.float64]
        """Provides the fractional coverage of observations for each year."""
        self.years: NDArray[np.datetime64]
        """The covered years as np.datetime64 at year precision."""
        self.year_total_seconds: NDArray[np.int_]
        """The total number of seconds for each year in the time series."""
        self.year_n_days: NDArray[np.floating]
        """The total number of days in each year in the time series."""
        self.year_n_growing_days: NDArray[np.floating]
        """The total number of growing days for each year in the time series. If the
        growing_season input varies within days, these values can contain non-integer
        values."""

        # Sanity checks on datetimes
        if not (
            isinstance(timing, AcclimationModel)
            or (
                isinstance(timing, np.ndarray)
                and np.issubdtype(timing.dtype, np.datetime64)
                and timing.ndim == 1
            )
        ):
            raise ValueError(
                "The timings argument must be an AcclimationModel "
                "or a one-dimensional array of datetime64 values"
            )

        if isinstance(timing, AcclimationModel):
            # AcclimationModel by construction provides subdaily data with equal spacing
            self.datetimes = timing.datetimes.astype("datetime64[s]")
            duration_last_observation = timing.spacing.astype("timedelta64")
        else:
            # Pure datetime inputs could be any frequency from subdaily to monthly, and
            # some frequencies could be of differing lengths (monthly being a good
            # example)

            # Convert time to seconds precision
            self.datetimes = timing.astype("datetime64[s]")

            # Get the intervals in seconds and see if they are strictly increasing and
            # then if the spaing is consistent. If the spacing is not consistent, then
            # require an endpoint for the observations.
            duration_seconds = np.diff(self.datetimes)

            if not np.all(duration_seconds > 0):
                raise ValueError("The timing values are not strictly increasing")

            intervals: NDArray = np.unique(duration_seconds)

            if len(intervals) == 1:
                # Constant intervals
                duration_last_observation = duration_seconds[0]
            else:
                if endpoint is None:
                    raise ValueError(
                        "The timings values are not equally spaced: provide an "
                        "explicit endpoint"
                    )

                if endpoint <= timing[-1]:
                    raise ValueError(
                        "The end_datetime value must be greater than the "
                        "last timing value"
                    )

                duration_last_observation = (endpoint - self.datetimes[-1]).astype(
                    "timedelta64[s]"
                )

        self.n_obs = self.datetimes.size

        # Sanity checks on growing season
        if growing_season is None:
            growing_season = np.ones_like(self.datetimes, dtype=np.bool_)
        else:
            if not np.issubdtype(growing_season.dtype, np.bool_):
                raise ValueError(
                    "Growing season data is not an array of boolean values"
                )

            if not self.datetimes.shape == growing_season.shape:
                raise ValueError(
                    "Growing season data is not the same shape as the timing data"
                )
        # Store the growing season data
        self.growing_season = growing_season

        # Record the endpoint to get the total timespan of the data and hence the
        # duration of each observation
        self.endpoint = self.datetimes[-1] + duration_last_observation
        timespan = np.append(self.datetimes, self.endpoint)
        self.duration_seconds = np.diff(timespan).astype(np.int_)

        # Now get the datetimes of the start of each year included in the data
        years = np.unique(timespan.astype("datetime64[Y]"))

        # Unless the last timespan value is exactly equal to the end of the previous
        # year, add the next year to the list of years to handle trailing data.
        # Also record the identity of each year in a way that handles years ending
        # exactly on the year change (and which therefore end with a year that has no
        # data)
        if not (years[-1] == timespan[-1]):
            self.years = np.copy(years)
            years = np.append(years, years[-1] + np.timedelta64(1, "Y"))
        else:
            self.years = np.copy(years[:-1])

        # Convert to second precision and find where they occur in the timespan
        years = years.astype("datetime64[s]")
        year_change_indices = np.searchsorted(timespan, years)

        # Now assign the duration of each observation across years, allowing for year
        # changes that occur during an observation, storing the indices of subsets and
        # the weighting to be used with values.

        # Iterate over pairs of year dates and indices
        for (lower, upper), (lower_index, upper_index) in zip(
            pairwise(years), pairwise(year_change_indices)
        ):
            # Get the initial set of datetimes within the year
            year_datetimes = timespan[lower_index:upper_index]

            # If the upper index is not to the end of the time series, then append the
            # closing time for the current year at the end and extend the sample to
            # include the next value.
            #
            # Note here that the indexing of the final observation does not require
            # special handling because np.searchsorted returns a last index _beyond_ the
            # end of the timespan, so will automatically include the last observation.
            if upper_index < len(timespan):
                year_datetimes = np.append(year_datetimes, upper)

            # If the first observation is not the precise start of the year _and_ we are
            # not on the first year of data, then we also need to shift lower_index down
            # to include partial data from the previous observation and add the year
            # start to the internal datetimes
            if (year_datetimes[0] != lower) and (lower_index > 0):
                lower_index -= 1
                year_datetimes = np.insert(year_datetimes, 0, lower)

            # Calculate the duration of the observations within the year span
            internal_year_durations = np.diff(year_datetimes).astype(np.int_)

            # Divide the internal duration through by the actual observation durations
            # to get fractional weights.
            fractional_duration = (
                internal_year_durations / self.duration_seconds[lower_index:upper_index]
            )

            # Store the indices and weights
            self.indexing.append((int(lower_index), int(upper_index)))
            self.duration_weights.append(internal_year_durations)
            self.fractional_weights.append(fractional_duration)

        # Split the growing season up into a list of subarrays by year
        self.growing_season_by_year = [
            growing_season[lower:upper] for lower, upper in self.indexing
        ]

        # Populate the year completeness and day counts
        self.year_total_seconds = np.diff(years).astype(np.int_)
        self.year_completeness = (
            np.array([np.sum(v) for v in self.duration_weights])
            / self.year_total_seconds
        )
        day_seconds = 86400
        self.year_n_days = self.year_total_seconds / day_seconds
        self.year_n_growing_days = np.array(
            [
                np.sum(x * y) / day_seconds
                for x, y in zip(self.duration_weights, self.growing_season_by_year)
            ]
        )

    def _split_values_by_year(
        self, values: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """Validates and splits value arrays.

        Args:
            values: An array of values.
        """

        if values.shape[0] == 1:
            # Broadcast to match the number of observations if constant
            values = np.broadcast_to(values, (self.n_obs, *values.shape[1:]))

        elif values.shape[0] != self.n_obs:
            raise ValueError(
                "First axis of values shape does not match number of observations."
            )

        # Split the daily values into subarrays for each year
        return [values[lower:upper] for lower, upper in self.indexing]

    def get_annual_means(
        self,
        values: NDArray[np.float64],
        within_growing_season: bool = False,
    ) -> NDArray[np.floating]:
        """Get annual means from an array of values.

        Average values are calculated weighted by the __duration__ of each observation,
        including weighting partial observations than span year boundaries. If
        ``within_growing_season`` is ``True``, the weights for observations outside of
        the observations marked as the growing season are set to zero. The calculation
        handles missing values.

        The annual means can be computed for a range of different sites by using n-D
        array for ``values``. In this case time is assumed along axis 0 and so an n-D
        array will be returned with annual values along axis 0.

        Example:
            >>> # Three years of monthly data
            >>> datetimes = np.arange(
            ...     np.datetime64('2000-01'),
            ...     np.datetime64('2003-01'),
            ...     np.timedelta64(1, "M")
            ... )
            >>> # Monthly data is uneven - requires an explicit endpoint.
            >>> avc = AnnualValueCalculator(
            ...     datetimes, endpoint=np.datetime64('2003-01')
            ... )
            >>> # Note that the means are weighted by the actual durations of months.
            >>> avc.get_annual_means(np.arange(0, 36)).round(4)
            array([ 5.5137, 17.526 , 29.526 ])

        Args:
            values: The data to summarize by year
            within_growing_season: Should the mean only include values within the
                growing season.
        """

        values_by_year = self._split_values_by_year(values)

        # Averages use _duration_ weights
        if within_growing_season:
            weights = [
                wght * gs
                for wght, gs in zip(self.duration_weights, self.growing_season_by_year)
            ]
        else:
            weights = self.duration_weights

        # Make weights broadcastable in case vals is multidimensional
        if values.ndim > 1:
            for i, wghts in enumerate(weights):
                shape = (wghts.shape[0],) + (1,) * (values.ndim - 1)
                weights[i] = wghts.reshape(shape)

        # Calculate the weighted mean in a np.nan friendly way: the product of np.nan
        # and a weight is np.nan and the isnan term omits the weights of nan
        # observations from the weighted average.
        # The mean is computed along just the time (0) axis.
        return np.array(
            [
                np.nansum(vals * wghts, axis=0)
                / np.nansum(~np.isnan(vals) * wghts, axis=0)
                for vals, wghts in zip(values_by_year, weights)
            ]
        )

    def get_annual_totals(
        self,
        values: NDArray[np.float64],
        within_growing_season: bool = False,
    ) -> NDArray[np.floating]:
        """Get annual totals from an array of values.

        The contribution of each observation to the total is weighted by the
        __fractional__ duration of each observation within each year in order to
        partition the sum across years correctly. If ``within_growing_season`` is
        ``True``, the weights for observations not identified as growing season values
        are set to zero. The method handles missing data (`np.nan`) but obviously the
        resulting annual total will be reduced.

        The annual totals can be computed for a range of different sites by using an n-D
        array for ``values``. In this case time is assumed along axis 0 and so an n-D
        array will be returned with annual values along axis 0.

        Example:
            >>> # Three years of monthly data with incomplete years at start and end
            >>> datetimes = np.arange(
            ...     np.datetime64('2000-07'),
            ...     np.datetime64('2003-07'),
            ...     np.timedelta64(1, "M")
            ... )
            >>> # Monthly data is uneven - requires an explicit endpoint.
            >>> avc = AnnualValueCalculator(
            ...     datetimes, endpoint=np.datetime64('2003-07')
            ... )
            >>> # Note that the means are weighted by the actual durations of months.
            >>> avc.get_annual_totals(np.arange(0, 36)).round(4)
            array([ 15., 138., 282., 195.])
            >>> # Year completeness: 184/366 days in 2000, 181/365 days in 2003.
            >>> avc.year_completeness.round(4)
            array([0.5027, 1.    , 1.    , 0.4959])

        Args:
            values: The data to summarize by year
            within_growing_season: Should the mean only include values within the
                growing season.
        """

        values_by_year = self._split_values_by_year(values)

        # Totals use _fractional_ weights
        if within_growing_season:
            weights = [
                wght * gs
                for wght, gs in zip(
                    self.fractional_weights, self.growing_season_by_year
                )
            ]
        else:
            weights = self.fractional_weights

        # Make weights broadcastable in case vals is multidimensional
        if values.ndim > 1:
            for i, wghts in enumerate(weights):
                shape = (wghts.shape[0],) + (1,) * (values.ndim - 1)
                weights[i] = wghts.reshape(shape)

        # The total is computed along just the time (0) axis.
        return np.array(
            [
                np.nansum(vals * wghts, axis=0)
                for vals, wghts in zip(values_by_year, weights)
            ]
        )


def broadcast_time(values: NDArray, shape: tuple[int, ...]) -> NDArray:
    """Broadcast an array along the time (zeroth) axis.

    The ``values`` array must be broadcastable to the full shape, however it does not
    need the full set of dimensions as defined by ``shape``. The returned array will
    have the full set of dimensions, and be broadcast along just the zeroth axis.

    Example:
        >>> broadcast_time(np.ones((1,3)), (2,3))
        array([[1., 1., 1.],
               [1., 1., 1.]])
        >>> broadcast_time(np.ones(3), (2,2,3)).shape
        (2, 1, 3)

    Args:
        values: The array to broadcast.
        shape: The full n-dimensional shape, where the first value is the length of the
            time axis to broadcast over.
    """
    if values.ndim > len(shape):
        raise ValueError("The input array has more dimensions than the broadcast shape")
    # Get any missing axes
    full_shape = (1,) * (len(shape) - len(values.shape)) + values.shape
    # Define the shape to broadcast to
    bcast_shape = (shape[0], *full_shape[1:])
    # Return the broadcasted array
    return np.broadcast_to(values, bcast_shape)
