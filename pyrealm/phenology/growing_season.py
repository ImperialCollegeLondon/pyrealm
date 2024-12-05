"""Tools to identify a growing season within annual cycles.

The growing season is identified as a number of days during which plants consistently
experience conditions above certain thresholds. A commonly used condition is days where
the temperature is greater than 0°C but any set of conditions could be used to generate
a binary sequence showing whether a day is suitable for growth or not.

The next step is then to address the issue of 'consistently' suitable for growth. This
module provides filters that can be used to remove short runs of days with unsuitable
conditions within a growing season and lead to more stable definitions of growing season
for a location.
"""

import numpy as np
from numpy.typing import NDArray

from pyrealm.core.calendar import Calendar


def filter_short_intervals_strides(
    ts: NDArray[np.bool_], window: int
) -> NDArray[np.bool_]:
    """TODO.

    Incomplete implementation - the aim here is to use strides to apply a filter across
    all axes simultaneously. The basic idea is to use as_strided to wrap an input array
    with a particular window size:

        [1,2,3,4,5,6]

        [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]

    That allows logical tests along that new axis to check if at least one day within a
    window has growing conditions and hence allow short runs to be identified. The basic
    structure is there, but it is tricky to finesse the details. The second version
    below is much easier to implement and so is an easier place to start - but it is
    likely to be much less efficient.

    Args:
        ts: An array of time series
        window: The maximum interval length to remove
        axis: The axis along which to filter intervals
    """

    raise NotImplementedError("The implementation on this function is incomplete.")

    # Draws from: https://stackoverflow.com/a/52219082/3401916

    # Generate the shape and strides required to wrap moving windows along the data onto
    # a new last axis. The shape simply adds on the extra dimension and the strides
    # repeats the current striding on the last axis for the new axis. The result
    shape = (*ts.shape, window + 1)
    strides = (*ts.strides, ts.strides[-1])
    forward = np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

    # TODO need to fix trailing values
    forward[-1, 1:] = 1

    # Find the sum of values along the new axis
    forward_sum = np.sum(forward, axis=1)

    reverse = np.lib.stride_tricks.as_strided(forward_sum, shape=shape, strides=strides)

    reverse_min = np.min(reverse, axis=1)

    return np.concatenate([np.ones(window - 1), reverse_min]) > 0


def filter_short_intervals(ts: NDArray[np.bool_], window: int) -> NDArray[np.bool_]:
    """Filter short intervals of unsuitable conditions within growing season data.

    This function takes an boolean array that contains a time series of observations on
    whether a particular day is suitable for growth. It identifies and replaces short
    runs of unsuitable conditions in order to give a cleaned approximation of the
    growing season within the time series.

    This implementation uses run length encoding to identify the length of runs of
    suitable and unsuitable conditions and then sets runs of unsuitable conditions that
    are less than or equal to the window length to be suitable. The time series is
    assumed to be along the first axis and the implementation iterates over all other
    dimensions.

    This is likely to be slow with large and/or multidimensional inputs.

    Args:
        ts: An array of time series
        window: The maximum interval length to remove
    """

    ts_filtered = np.empty_like(ts)

    # Generate an iterator over axes other than the first
    for idx in np.ndindex(ts[0].shape):
        # Pull out the slice along the first axis and run length encode it
        run_lengths, run_values = run_length_encode(ts[:, *idx])  # type: ignore[arg-type]

        # Get rid of runs of zeros below the window length
        short_window = np.logical_and(run_values == 0, run_lengths <= window)
        run_values = np.where(short_window, 1, run_values)

        # Insert filtered values
        ts_filtered[:, *idx] = np.repeat(run_values, run_lengths)

    return ts_filtered


def run_length_encode(values: NDArray) -> tuple[NDArray[np.int_], NDArray]:
    """Calculate run length encoding of 1D arrays.

    The function returns a tuple containing an array of the run lengths and an array of
    the values of each run. These can be turned back into the original array using
    ``np.repeat(values, run_lengths)``.

    Args:
        values: A one dimensional array of values
    """

    n = values.size
    if n == 0 or values.ndim != 1:
        raise ValueError(
            "run_length_encode requires a 1 dimensional array of non-zero length"
        )

    # Find where adjacent values are not equal
    pairs_not_equal = values[1:] != values[:-1]
    # Find change points where values are not equal and add last position.
    change_points = np.append(np.where(pairs_not_equal), n - 1)
    # Get run lengths between change points
    run_lengths = np.diff(np.append(-1, change_points))

    return (run_lengths, values[change_points])


def find_cumulative_suitable_days(
    suitable: NDArray[np.bool_],
) -> NDArray[np.int_]:
    """Calculate runs of consecutive suitable days.

    This functions takes an boolean array representing a daily time series of whether
    growing conditions are suitable for each day. It calculates the cumulative sum of
    the number of consecutive days of suitable growth, resetting the sum to zero when an
    unsuitable day is encountered.

    Examples:
        >>> suitable = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0])
        >>> accumulate_consecutive_suitable_days(suitable)
        [0, 0, 1, 2, 3, 0, 1, 2, 0]

    Args:
        suitable: A boolean array of growth suitability data.
    """
    # Calculate a cumulative sum of suitable days
    cumulative_suitable_days = np.cumsum(suitable, axis=0, dtype="int")

    # Find the cumulative sum of consecutive suitable days. The accumulation resets
    # the count to zero when an unsuitable day is encountered.
    consecutive_suitable_days = cumulative_suitable_days - np.maximum.accumulate(
        np.where(np.logical_not(suitable), cumulative_suitable_days, 0), axis=0
    )

    return consecutive_suitable_days


def find_annual_growing_season(
    suitable: NDArray[np.bool_],
    dates: NDArray[np.datetime64],
    return_dates: bool = False,
) -> (
    NDArray[np.int_]
    | tuple[NDArray[np.datetime64], NDArray[np.datetime64], NDArray[np.int_]]
):
    """Find the growing season from annual suitability data.

    This function takes an boolean array representing a daily time series for a single
    year of whether growing conditions are suitable for each day. It calculates the
    length of the maximum run of suitable days and, optionally, also returns the start
    and end dates of the growing season.

    Args:
        suitable: A boolean array of growth suitability data.
        dates: A numpy array of dates as ``np.datetime64``.
        return_dates: Should the function return the start and end dates of the season
            as well as the season length.
    """

    # Find the cumulative consecutive suitable days for the year
    year_consecutive = find_cumulative_suitable_days(suitable)

    # Find the index of the maximum and use it to extract the season length and the
    # season start and end dates
    year_max_consecutive_idx = np.argmax(year_consecutive, axis=0, keepdims=True)
    season_length = np.take_along_axis(
        year_consecutive, year_max_consecutive_idx, axis=0
    ).squeeze()

    # Just return the season lengths
    if not return_dates:
        return season_length

    # Calculate the season start and end dates if requested
    season_end = dates[year_max_consecutive_idx].squeeze()
    season_start = dates[year_max_consecutive_idx - season_length + 1].squeeze()

    return season_start, season_end, season_length


def find_growing_seasons(
    suitable: NDArray[np.bool_],
    dates: NDArray[np.datetime64],
    return_dates: bool = False,
) -> (
    NDArray[np.int_]
    | tuple[NDArray[np.datetime64], NDArray[np.datetime64], NDArray[np.int_]]
):
    """Find growing seasons across multiple years.

    This function takes an boolean array representing a multiple year time series of
    daily data showing whether growing conditions are suitable for each day. It
    calculates the length of the maximum run of suitable days within each year and,
    optionally, also returns the start and end dates of the growing season.

    Args:
        suitable: A boolean array of growth suitability data.
        dates: A numpy array of dates as ``np.datetime64``.
        return_dates: Should the function return the start and end dates of the season
            as well as the season length.
    """
    # TODO need to add hemisphere switching

    # Get the dates as a Calendar object
    calendar = Calendar(dates)

    if dates.shape != suitable[0].shape:
        raise ValueError("Number of dates not equal to first axis length of suitable")

    # Split the input arrays along the first axis to give blocks of suitability and
    # dates for each year
    year_change_indices = np.where(np.diff(calendar.year) > 0)[0]
    year_subarrays = np.split(suitable, year_change_indices)
    date_subarrays = np.split(dates, year_change_indices)

    # Calculate the annual growing season within each year split
    seasons = [
        find_annual_growing_season(
            suitable=year_array, dates=date_array, return_dates=return_dates
        )
        for year_array, date_array in zip(year_subarrays, date_subarrays)
    ]

    # Repackage into arrays

    return seasons
