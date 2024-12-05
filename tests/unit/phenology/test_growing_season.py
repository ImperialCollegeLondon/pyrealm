"""Test the implementation used for demarcating growing season."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal


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

        assert_allclose(result, expected)


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
        assert_allclose(input, np.repeat(run_values, run_lengths))
        return

    assert str(err.value) == err_msg


@pytest.mark.parametrize(
    argnames="ts_reshape, ts_broadcast",
    argvalues=[
        pytest.param((21,), (21,), id="row_vector"),
        pytest.param((21, 1), (21, 1), id="column_vector"),
        pytest.param((21, 1), (21, 2), id="2D"),
        pytest.param((21, 1, 1), (21, 2, 2), id="3D"),
    ],
)
def test_find_cumulative_suitable_days(ts_reshape, ts_broadcast):
    """Test find_cumulative_suitable_days.

    Uses a simple test case to check calculation works across inputs of differing
    dimensions
    """
    from pyrealm.phenology.growing_season import find_cumulative_suitable_days

    inputs = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    expected = np.array([0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 0, 1, 0])

    # Broadcast the data into different dimensionalities
    inputs_reshape = np.broadcast_to(inputs.reshape(ts_reshape), ts_broadcast)
    expected_reshape = np.broadcast_to(expected.reshape(ts_reshape), ts_broadcast)

    # Check the calculation works
    calculated = find_cumulative_suitable_days(inputs_reshape)

    assert_allclose(calculated, expected_reshape)


@pytest.mark.parametrize(
    argnames="ts_reshape, ts_broadcast",
    argvalues=[
        pytest.param((21,), (21,), id="row_vector"),
        pytest.param((21, 1), (21, 1), id="column_vector"),
        pytest.param((21, 1), (21, 2), id="2D"),
        pytest.param((21, 1, 1), (21, 2, 2), id="3D"),
    ],
)
def test_find_annual_growing_season_dimensions(ts_reshape, ts_broadcast):
    """Test find_annual_growing_season.

    Uses a simple short test case to check calculation works across inputs of differing
    dimensions.
    """
    from pyrealm.phenology.growing_season import find_annual_growing_season

    inputs = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    dates = np.arange(
        np.datetime64("2024-06-01"), np.datetime64("2024-06-21"), np.timedelta64(1, "D")
    )

    # Broadcast the data into different dimensionalities
    inputs_reshape = np.broadcast_to(inputs.reshape(ts_reshape), ts_broadcast)
    expected_season_length = np.full_like(inputs_reshape[0], 4)
    expected_season_start = np.full_like(
        inputs_reshape[0], np.datetime64("2024-06-04"), dtype="datetime64[D]"
    )
    expected_season_end = np.full_like(
        inputs_reshape[0], np.datetime64("2024-06-07"), dtype="datetime64[D]"
    )

    # Check the calculation works
    season_start, season_end, season_length = find_annual_growing_season(
        inputs_reshape, dates, return_dates=True
    )

    # All integers and dates so use assert_equal
    assert_equal(season_length, expected_season_length)
    assert_equal(season_start, expected_season_start)
    assert_equal(season_end, expected_season_end)


def test_find_annual_growing_season_complex():
    """Test find_annual_growing_season.

    Uses a more 3D complex input where the results should differ across the cells.
    """
    from pyrealm.phenology.growing_season import find_annual_growing_season

    # Construct 4 time series

    dates = np.arange(
        np.datetime64("2021-01-01"), np.datetime64("2022-01-01"), np.timedelta64(1, "D")
    )

    growing_seasons = (
        (np.datetime64("2021-06-01"), np.datetime64("2021-07-31")),
        (np.datetime64("2021-05-01"), np.datetime64("2021-08-31")),
        (np.datetime64("2021-04-01"), np.datetime64("2021-09-30")),
        (np.datetime64("2021-03-01"), np.datetime64("2021-10-31")),
    )

    # Fill in ones between those dates on each time series in a 2x2 grid
    inputs = np.zeros((365, 2, 2))
    for (grow_start, grow_end), idx in zip(
        growing_seasons, np.ndindex(inputs[0].shape)
    ):
        inputs[
            slice(
                np.where(dates == grow_start)[0][0], np.where(dates == grow_end)[0][0]
            ),
            *idx,
        ] = 1

    # Get the expectations
    expected_season_length = np.array(
        [(b - a) / np.timedelta64(1, "D") for a, b in growing_seasons], dtype="int"
    ).reshape((2, 2))

    growing_seasons_array = np.array(growing_seasons).reshape((2, 2, 2))
    expected_season_start = growing_seasons_array[:, :, 0]
    expected_season_end = growing_seasons_array[:, :, 1] - np.timedelta64(1, "D")

    # Check the calculation works
    season_start, season_end, season_length = find_annual_growing_season(
        inputs, dates, return_dates=True
    )

    # All integers and dates so use assert_equal
    assert_equal(season_length, expected_season_length)
    assert_equal(season_start, expected_season_start)
    assert_equal(season_end, expected_season_end)


@pytest.mark.parametrize(
    argnames="inputs, outcome, expected",
    argvalues=(
        pytest.param(
            np.zeros(365),
            does_not_raise(),
            ("NaT", "NaT", 0),
            id="no_grow",
        ),
        pytest.param(
            np.ones(365),
            does_not_raise(),
            ("2021-01-01", "2021-12-31", 365),
            id="all_grow",
        ),
        pytest.param(
            np.ones(366),
            pytest.raises(ValueError),
            None,
            id="dates_unequal",
        ),
    ),
)
def test_find_annual_growing_season_edges_and_exceptions(inputs, outcome, expected):
    """Test edge case and exception handling in find_annual_growing_season."""
    from pyrealm.phenology.growing_season import find_annual_growing_season

    dates = np.arange(
        np.datetime64("2021-01-01"), np.datetime64("2022-01-01"), np.timedelta64(1, "D")
    )

    with outcome:
        season_start, season_end, season_length = find_annual_growing_season(
            inputs, dates, return_dates=True
        )

        # Test assertions if no error
        assert_equal(
            season_start, np.array(np.datetime64(expected[0]), dtype="datetime64[D]")
        )
        assert_equal(
            season_end, np.array(np.datetime64(expected[1]), dtype="datetime64[D]")
        )
        assert_equal(season_length, np.array(expected[2]))
