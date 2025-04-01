"""Testing the annual value calculator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="function, within_growing_season, expected",
    argvalues=(
        pytest.param(
            "mean",
            True,
            np.repeat(1, 10),
            id="mean",
        ),
        pytest.param(
            "sum",
            True,
            np.repeat(200, 10),
            id="sum",
        ),
        pytest.param(
            "sum",
            False,
            np.array([366, 365, 365, 365, 366, 365, 365, 365, 366, 365]),
            id="sum",
        ),
    ),
)
def test_AnnualValueCalculator(function, within_growing_season, expected):
    """Basic tests of the AnnualValueCalculator.

    This checks that the annual value calculator extracts the 1 values successfully from
    within the daily data and then correctly summarises them for different functions and
    with and without the growing season.
    """

    from pyrealm.phenology.fapar_limitation import AnnualValueCalculator

    # Set up a 10 year period
    datetimes = np.arange(
        np.datetime64("2000-01-01"),
        np.datetime64("2010-01-01"),
        np.timedelta64(30, "m"),
    )

    # Set up a growing season that includes 200 days in the middle of each year
    _, obs_in_year = np.unique(datetimes.astype("datetime64[Y]"), return_counts=True)
    days_in_year = obs_in_year / 48
    growing_season = np.concatenate(
        [np.repeat([False, True, False], (75, 200, n - 275)) for n in days_in_year]
    )

    # Values - has ones at midday for all days
    values = np.tile(np.repeat((0, 1, 0), (24, 1, 23)), 3653)

    # Create the calculator instance
    avc = AnnualValueCalculator(datetimes=datetimes, growing_season=growing_season)

    # Apply the method and check the outcome.
    result = avc.get_annual_values(
        values=values, function=function, within_growing_season=within_growing_season
    )

    assert_allclose(result, expected)
