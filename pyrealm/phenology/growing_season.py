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
