"""Test the implementation used for demarcating growing season."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="ts_reshape, ts_broadcast",
    argvalues=[
        pytest.param((65,), (65,), id="row_vector"),
        pytest.param((65, 1), (65, 1), id="column_vector"),
        pytest.param((65, 1), (65, 2), id="2D"),
        pytest.param((65, 1, 1), (65, 2, 2), id="3D"),
    ],
)
def test_filter_short_intervals(ts_reshape, ts_broadcast):
    """Test filter_short_intervals."""

    from pyrealm.phenology.growing_season import filter_short_intervals

    # Define a time series with some zero windows of varying length
    ts = np.ones(65)
    windows = ((5,), (15, 16), (25, 26, 27), (35, 36, 37, 38), (45, 46, 47, 48, 49))
    for win in windows:
        ts[[win]] = 0

    # Broadcast the data into different dimensionalities
    ts = np.broadcast_to(ts.reshape(ts_reshape), ts_broadcast)

    # Loop over a range of window lengths
    for win_length in np.arange(1, 6):
        # Run the filter
        result = filter_short_intervals(ts=ts, window=win_length)

        # Build and check the expected result - window lengths greater than the current
        # win_length should be unaffected, so make an array of ones and then fill in the
        # zeros for the longer windows
        expected = np.ones_like(ts)

        # For all windows with an index greater than the current window length
        for win in windows[win_length:]:
            expected[[win], ...] = 0

        assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="input, outcome, err_msg",
    argvalues=[
        pytest.param(
            np.array([]),
            pytest.raises(ValueError),
            "run_length_encode requires a 1 dimensional array of non-zero length",
            id="zero length",
        ),
        pytest.param(
            np.ones((10, 2)),
            pytest.raises(ValueError),
            "run_length_encode requires a 1 dimensional array of non-zero length",
            id="not 1D",
        ),
        pytest.param(
            np.random.binomial(1, 0.5, size=(100,)),
            does_not_raise(),
            None,
            id="pass_boolean",
        ),
        pytest.param(
            np.random.choice(np.arange(6), size=(100,)),
            does_not_raise(),
            None,
            id="pass_int",
        ),
    ],
)
def test_run_length_encode(input, outcome, err_msg):
    """Test run length encode."""
    from pyrealm.phenology.growing_season import run_length_encode

    with outcome as err:
        run_lengths, run_values = run_length_encode(input)

        # Check that the encoding can be used to reconstruct the original.
        assert np.allclose(input, np.repeat(run_values, run_lengths))
        return

    assert str(err.value) == err_msg
