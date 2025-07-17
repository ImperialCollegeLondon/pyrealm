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
    """Annual means and totals from time series data.

    This class calculates annual means and totals from time series data and is designed
    to handle time series data sampled at a wide range of intervals. If time series are
    sampled at intervals that do not map neatly onto years - such as weekly or
    fortnightly data - or do not have precisely uneven sampling - such as monthly data,
    then calculating annual means and sums becomes awkward.

    This class handles the issue by taking a set of datetimes for observations and
    mapping them onto years. Observations that span year boundaries are included in both
    years and the class calculates two forms of weightings:

    * Duration weights provide the actual duration of each observation within a year,
      and are used to calculate weighted means
    * Proportion weights provide the proportion of an observation in a given year and
      are used to calculate annual sums.

    The class must be created for a specific target dataset with a known array shape
    (the ``data_shape`` argument), with the first axis representing time. The
    ``timings`` argument then sets the timing of observations along that axis. Lastly,
    the optional ``subset_mask`` argument can be used to define a subset of observations
    to be used when calculating totals and means. This subset mask must be an array that
    can be broadcast to the data shape. The datetimes do not have to completely sample
    all years: the ``year_completeness`` attribute records what fraction of a year is
    covered by the datetimes.

    As an example, 10 years of monthly data for a 5 x 5 spatial grid of sites could
    have:

    * a data shape of ``(120, 5, 5)``,
    * a timings array with shape ``(120,)``, and then
    * a subset mask with shape ``(120, 1, 1)``, providing a single subset mask for use
      at all 25  grid cells. Alternatively, a subset mask with shape ``(120, 5, 5)``
      would provide site specific subsets.

    With uneven sampling - such as monthly - an explicit endpoint must be provided to
    set the duration of the final observation.

    Once an instance is created, then the
    :meth:`AnnualValueCalculator.get_annual_totals` and
    :meth:`AnnualValueCalculator.get_annual_means` methods can be used to calculate the
    actual summary statistics for arrays of values that match the data shape.

    The ``timings`` argument can either directly provide an array of
    :class:`numpy.datetime64` values, or use an existing
    :class:`~pyrealm.pmodel.acclimation.AcclimationModel` object, which contains such an
    array. The ``subset_mask`` argument must be a boolean array.

    Example:
        >>> # Three years of monthly data - 36 observations
        >>> datetimes = np.arange(
        ...     np.datetime64('2000-01'),
        ...     np.datetime64('2003-01'),
        ...     np.timedelta64(1, "M")
        ... )
        >>> # Monthly data is uneven - requires an explicit endpoint.
        >>> avc = AnnualValueCalculator(
        ...    data_shape=(36,),
        ...    timing=datetimes,
        ...    endpoint=np.datetime64('2003-01')
        ... )
        >>> avc.year_completeness
        array([1., 1., 1.])

    Args:
        timing: A :class:`~pyrealm.pmodel.acclimation.AcclimationModel` instance or a
            one-dimensional array of :class:`numpy.datetime64` values.
        data_shape: A tuple giving the shape of the data to be used with an instance.
        subset: An array of :class:`numpy.bool_` values to be used in excluding
            observations from summary stat calculations within a year.
        endpoint: A single :class:`numpy.datetime64` value, required to provide an
            explicit endpoint for observations with uneven frequency.
    """

    def __init__(
        self,
        data_shape: tuple[int, ...],
        timing: AcclimationModel | NDArray[np.datetime64],
        subset_mask: NDArray[np.bool_] | None = None,
        endpoint: np.datetime64 | None = None,
    ):
        # Attribute definitions
        self.data_shape: tuple[int, ...]
        """The shape of data value arrays accepted by an instance."""
        self.datetimes: NDArray[np.datetime64]
        """The start datetime of observations taking from the initial timings"""
        self.n_obs: int
        """The number of observations in the time series."""
        self.endpoint: np.datetime64
        """A datetime giving of the end of the last observation."""
        self.subset_mask: NDArray[np.bool_]
        """The initial input array for the subset mask."""
        self.duration_seconds: NDArray[np.int_]
        """The duration of each observation in seconds."""
        self.indexing: list[tuple[int, int]] = []
        """Pairs of integers giving start and end indices to extract consecutive years
        of data from the time series."""
        self.duration_weights: list[NDArray[np.int_]] = []
        """A list of arrays giving the number of seconds that each observation
        within a year contributes to that year."""
        self.fractional_weights: list[NDArray[np.floating]] = []
        """A list of arrays giving the fraction of each observation within a year that
        falls in the year."""
        self.subset_mask_by_year: list[NDArray[np.bool_]]
        """A list of arrays giving the subset mask subarrays for each year."""
        self.year_completeness: NDArray[np.floating]
        """Provides the fractional coverage of observations for each year."""
        self.years: NDArray[np.datetime64]
        """The covered years as np.datetime64 at year precision."""
        self.year_total_seconds: NDArray[np.int_]
        """The total number of seconds for each year in the time series."""
        self.year_n_days: NDArray[np.float64]
        """The total number of days in each year in the time series."""
        self.year_n_growing_days: NDArray[np.float64]
        """The total number of growing days for each year in the time series. If the
        growing_season input varies within days, these values can contain non-integer
        values."""

        # Check data shape
        if not (
            isinstance(data_shape, tuple)
            and all(isinstance(v, int) for v in data_shape)
        ):
            raise ValueError("The data_shape argument must a tuple of integers")

        self.data_shape = data_shape

        # Calculate a list of additional dimensions that can be used to broadcast 1D
        # arrays to the data shape
        extra_dims = [np.newaxis] * (len(self.data_shape) - 1)

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

            intervals: NDArray[np.floating] = np.unique(duration_seconds)

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

        # Get the number of observations and check it matches the declared data shape
        self.n_obs = self.datetimes.size
        if self.n_obs != self.data_shape[0]:
            raise ValueError(
                "The number of observation timings does not match the first "
                "axis of the data shape"
            )

        # Sanity checks on subset
        if subset_mask is None:
            # Default is an array with the same dimensions as the data shape, but with
            # singleton dimensions on all but time axis: broadcasts across shape
            mask_shape = [1] * len(self.data_shape)
            mask_shape[0] = self.n_obs
            subset_mask = np.ones(mask_shape, dtype=np.bool_)

        # Must be boolean
        if not np.issubdtype(subset_mask.dtype, np.bool_):
            raise ValueError("Subset mask data is not an array of boolean values")

        # Must be congruent with the data shape
        try:
            np.broadcast_shapes(subset_mask.shape, self.data_shape)
        except ValueError:
            raise ValueError(
                f"The subset mask shape {subset_mask.shape} and "
                f"data shape {self.data_shape} are not congruent"
            )

        # Store the growing season data, which is now not None
        self.subset_mask = subset_mask

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

            # Lastly, make the weights broadcastable to the data shape and store them
            self.duration_weights.append(internal_year_durations[:, *extra_dims])
            self.fractional_weights.append(fractional_duration[:, *extra_dims])

        # Split the subset mask up into a list of subarrays by year
        self.subset_mask_by_year = [
            self.subset_mask[lower:upper] for lower, upper in self.indexing
        ]

        # Populate the year completeness and day counts
        self.year_total_seconds = np.diff(years).astype(np.int_)
        self.year_completeness = (
            np.array([np.sum(v) for v in self.duration_weights])
            / self.year_total_seconds
        )

        # Calculate the year length in days and the number of days within the subset
        # mask per year. The first one _has_ to be constant across extra dimensions, but
        # the subset mask could be variable. Having different dimensionality on these
        # two attributes is confusing, so broadcast year_n_days to the data shape.
        day_seconds = 86400

        self.year_n_days = (self.year_total_seconds / day_seconds)[:, *extra_dims]

        self.year_n_days_subset = np.array(
            [
                np.sum(wght * mask, axis=0) / day_seconds
                for wght, mask in zip(self.duration_weights, self.subset_mask_by_year)
            ]
        )

    def _split_values_by_year(
        self, values: NDArray[np.floating]
    ) -> list[NDArray[np.floating]]:
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
        values: NDArray[np.floating],
        within_subset: bool = False,
    ) -> NDArray[np.floating]:
        """Get annual means from an array of values.

        Annual mean values are calculated using a weighted mean of values falling within
        each year, using the total __duration__ of each observation within the year as a
        weight. If an observation spans two years, the observation is included in the
        mean for both years, weighted by the total duration within the year. The method
        handles missing data (`np.nan`).

        The input values must be an array with a shape that matches the ``data_shape``
        used to create the :class:`AnnualValueCalculator` instance. Annual means are
        calculated along the first axis of the input array and array dimensions are
        preserved along all but that first axis. For example, monthly data values over
        10 years for a 3 x 3 grid of cells might have shape `(120, 3, 3)` - the
        resulting array of mean values would have shape `(10, 3, 3)`.

        The ``within_subset`` argument can be used to calculate mean values only from
        observations included in the ``subset_mask`` defined when creating the
        :class:`AnnualValueCalculator` instance.

        .. NOTE::

            The method returns values for incomplete years, but obviously these may be
            be biased as a result.

        Example:
            >>> # Three years of monthly data
            >>> datetimes = np.arange(
            ...     np.datetime64('2000-01'),
            ...     np.datetime64('2003-01'),
            ...     np.timedelta64(1, "M")
            ... )
            >>> # Monthly data is uneven - requires an explicit endpoint.
            >>> avc = AnnualValueCalculator(
            ...    data_shape=(36,),
            ...    timing=datetimes,
            ...    endpoint=np.datetime64('2003-01')
            ... )
            >>> # Note that the means are weighted by the actual durations of months.
            >>> avc.get_annual_means(np.arange(0, 36)).round(4)
            array([ 5.5137, 17.526 , 29.526 ])

        Args:
            values: The data to summarize by year
            within_subset: Should the mean only include values within the subset mask.
        """

        # Enforce shape
        if values.shape != self.data_shape:
            raise ValueError(
                f"Input values shape {values.shape} does not match "
                f"configured data shape {self.data_shape}"
            )

        values_by_year = self._split_values_by_year(values)

        # Averages use _duration_ weights
        if within_subset:
            weights = [
                wght * sub
                for wght, sub in zip(self.duration_weights, self.subset_mask_by_year)
            ]
        else:
            weights = self.duration_weights

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
        values: NDArray[np.floating],
        within_growing_season: bool = False,
        within_subset: bool = False,
    ) -> NDArray[np.floating]:
        """Get annual totals from an array of values.

        Annual totals are calculated by splitting the values into subarrays of values by
        year and then calculating the sum along the first axis. When observations span
        two years, they are included in both sets of annual data, but the sum is
        weighted by the fraction of the observation in each year to partition the value
        between years. The method handles missing data (`np.nan`) but obviously the
        resulting annual total will be reduced.

        The input values must be an array with a shape that matches the ``data_shape``
        used to create the :class:`AnnualValueCalculator` instance. Annual totals are
        calculated along the first axis of the input array and array dimensions are
        preserved along all but that first axis. For example, monthly data values over
        10 years for a 3 x 3 grid of cells might have shape `(120, 3, 3)` - the
        resulting array of total values would have shape `(10, 3, 3)`.

        The ``within_subset`` argument can be used to calculate totals values only from
        observations included in the ``subset_mask`` defined when creating the
        :class:`AnnualValueCalculator` instance.

        .. NOTE::

            The method returns values for incomplete years, but obviously these will
            be an underestimate. A simple correction would be to divide the totals by
            the year completeness to scale back up to full annual values.

        Example:
            >>> # Three years of monthly data with incomplete years at start and end
            >>> datetimes = np.arange(
            ...     np.datetime64('2000-07'),
            ...     np.datetime64('2003-07'),
            ...     np.timedelta64(1, "M")
            ... )
            >>> # Monthly data is uneven - requires an explicit endpoint.
            >>> avc = AnnualValueCalculator(
            ...    data_shape=(36,),
            ...    timing=datetimes,
            ...    endpoint=np.datetime64('2003-07')
            ... )
            >>> # Note that the means are weighted by the actual durations of months.
            >>> avc.get_annual_totals(np.arange(0, 36)).round(4)
            array([ 15., 138., 282., 195.])
            >>> # Year completeness: 184/366 days in 2000, 181/365 days in 2003.
            >>> avc.year_completeness.round(4)
            array([0.5027, 1.    , 1.    , 0.4959])

        Args:
            values: The data to summarize by year
            within_growing_season: Which values are within the growing season.
            within_subset: Should the mean only include values within the subset mask.
        """

        # Enforce shape
        if values.shape != self.data_shape:
            raise ValueError(
                f"Input values shape {values.shape} does not match "
                f"configured data shape {self.data_shape}"
            )

        values_by_year = self._split_values_by_year(values)

        # Totals use _fractional_ weights
        if within_subset:
            weights = [
                wght * gs
                for wght, gs in zip(self.fractional_weights, self.subset_mask_by_year)
            ]
        else:
            weights = self.fractional_weights

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
