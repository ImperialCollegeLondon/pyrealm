"""Utility functions for phenology calculations."""

import numpy as np
from numpy.typing import NDArray


# Local RLE function
def run_length_encode(
    values: NDArray[np.bool_ | np.int_],
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
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
