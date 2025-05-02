"""Testing the annual value calculator."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray


@pytest.mark.parametrize(
    argnames="datetimes, growing_season,  as_acclim, context_manager, error_message",
    argvalues=(
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(30, "m"),
            ),
            np.ones(
                (365 * 7 + 366 * 3) * 48, dtype=np.bool_
            ),  # 10 years of 30 min obs, with 3 leap years
            True,
            does_not_raise(),
            None,
            id="acclim_model_good",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(30, "m"),
            ),
            np.ones(
                (365 * 7 + 366 * 3) * 48, dtype=np.bool_
            ),  # 10 years of 30 min obs, with 3 leap years
            False,
            does_not_raise(),
            None,
            id="datetimes_half_hourly_good",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(30, "m"),
            ).astype(np.int_),
            np.ones(
                (365 * 5 + 366 * 3) * 48, dtype=np.bool_
            ),  # 10 years of 30 min obs, with 3 leap years
            False,
            pytest.raises(ValueError),
            "The timings argument must be an AcclimationModel "
            "or an array of datetime64 values",
            id="timings_not_acclim_or_datetimes",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(30, "m"),
            ),
            np.ones(
                (365 * 5 + 366 * 3) * 48, dtype=np.bool_
            ),  # 10 years of 30 min obs, with 3 leap years
            False,
            pytest.raises(ValueError),
            "Growing season data is not the same shape as the timing data",
            id="bad_growing_season_shape",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(30, "m"),
            ),
            np.ones(
                (365 * 7 + 366 * 3) * 48,
            ),  # 10 years of 30 min obs, with 3 leap years
            False,
            pytest.raises(ValueError),
            "Growing season data is not an array of boolean values",
            id="bad_growing_season_dtype",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-31"),
                np.datetime64("2009-11-30"),
                np.timedelta64(30, "m"),
            ),
            np.ones(
                (365 * 7 + 366 * 3) * 48, dtype=np.bool_
            ),  # 10 years of 30 min obs, with 3 leap years
            True,
            pytest.raises(ValueError),
            "Data timings do not cover complete years to within tolerance",
            id="acclim_model_incomplete_years",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(1, "D"),
            ),
            np.ones(
                (365 * 7 + 366 * 3), dtype=np.bool_
            ),  # 10 years of daily obs, with 3 leap years
            False,
            does_not_raise(),
            None,
            id="datetimes_daily_good",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(1, "W"),
            ),
            np.ones(522, dtype=np.bool_),  # 10 years of weekly obs
            False,
            does_not_raise(),
            None,
            id="datetimes_weekly_good",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-05"),
                np.datetime64("2010-01-01"),
                np.timedelta64(1, "W"),
            ),
            np.ones(522, dtype=np.bool_),  # 10 years of weekly obs
            False,
            does_not_raise(),
            None,
            id="datetimes_weekly_short_but_inside_tolerance_good",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-08"),
                np.datetime64("2010-01-01"),
                np.timedelta64(1, "W"),
            ),
            np.ones(522, dtype=np.bool_),  # 10 years of weekly obs
            False,
            does_not_raise(),
            None,
            id="datetimes_weekly_short_outside_tolerance",
        ),
    ),
)
def test_AnnualValueCalculator_init(
    datetimes, as_acclim, growing_season, context_manager, error_message
):
    """Test failure modes and success modes for initialising AVC instances."""
    from pyrealm.phenology.fapar_limitation import AnnualValueCalculator
    from pyrealm.pmodel.acclimation import AcclimationModel

    if as_acclim:
        datetimes = AcclimationModel(datetimes=datetimes)

    with context_manager as cmgr:
        _ = AnnualValueCalculator(timing=datetimes, growing_season=growing_season)
        return

    assert str(cmgr.value) == error_message


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
    avc = AnnualValueCalculator(timing=datetimes, growing_season=growing_season)

    # Apply the method and check the outcome.
    result = avc.get_annual_values(
        values=values, function=function, within_growing_season=within_growing_season
    )

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="datetimes, growing_season,  as_acclim, endpoint,"
    " context_manager, error_message, year_completeness",
    argvalues=(
        pytest.param(
            "a",
            None,
            False,
            None,
            pytest.raises(ValueError),
            "The timings argument must be an AcclimationModel "
            "or an array of datetime64 values",
            None,
            id="timings_not_acclim_or_datetimes",
        ),
        pytest.param(
            np.concat(  # 10 years of 30 min obs with duplicate to give 0s duration
                [
                    [np.datetime64("2000-01-01 00:00:00")],
                    np.arange(
                        np.datetime64("2000-01-01"),
                        np.datetime64("2010-01-01"),
                        np.timedelta64(30, "m"),
                    ),
                ]
            ),
            np.ones((365 * 5 + 366 * 3) * 48 + 1, dtype=np.bool_),
            False,
            None,
            pytest.raises(ValueError),
            "The timing values are not strictly increasing",
            None,
            id="timings_not_strictly_increasing",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01"),
                np.datetime64("2010-01"),
                np.timedelta64(1, "M"),
            ),
            np.ones(
                (365 * 5 + 366 * 3) * 48, dtype=np.bool_
            ),  # 10 years of 30 min obs, with 3 leap years with a duplicate at start
            False,
            None,
            pytest.raises(ValueError),
            "The timings values are not equally spaced: provide an explicit endpoint",
            None,
            id="unequal_no_endpoint",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01"),
                np.datetime64("2010-01"),
                np.timedelta64(1, "M"),
            ),
            np.ones(
                (365 * 5 + 366 * 3) * 48, dtype=np.bool_
            ),  # 10 years of 30 min obs, with 3 leap years with a duplicate at start
            False,
            np.datetime64("2005-06"),
            pytest.raises(ValueError),
            "The end_datetime value must be greater than the last timing value",
            None,
            id="endpoint_not_after_timings",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(30, "m"),
            ),
            np.ones(
                (365 * 5 + 366 * 3) * 48, dtype=np.bool_
            ),  # 10 years of 30 min obs, with 3 leap years
            False,
            None,
            pytest.raises(ValueError),
            "Growing season data is not the same shape as the timing data",
            None,
            id="bad_growing_season_shape",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(30, "m"),
            ),
            np.ones(
                (365 * 7 + 366 * 3) * 48,
            ),  # 10 years of 30 min obs, with 3 leap years
            False,
            None,
            pytest.raises(ValueError),
            "Growing season data is not an array of boolean values",
            None,
            id="bad_growing_season_dtype",
        ),
        pytest.param(
            np.arange(  # 10 years of fortnightly obs
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(2, "W"),
            ),
            None,
            False,
            None,
            does_not_raise(),
            None,
            np.concat([np.ones(10), [1 / 365]]),
            id="fortnightly",
        ),
        pytest.param(
            np.arange(  # 10 years of monthly obs
                np.datetime64("2000-01"),
                np.datetime64("2010-01"),
                np.timedelta64(1, "M"),
            ),
            None,
            False,
            np.datetime64("2010-01"),
            does_not_raise(),
            None,
            np.ones(10),
            id="monthly",
        ),
        pytest.param(
            np.arange(  # 10 years of fortnightly obs but offset by half a year
                np.datetime64("2000-06-01"),
                np.datetime64("2010-06-01"),
                np.timedelta64(2, "W"),
            ),
            None,
            False,
            None,
            does_not_raise(),
            None,
            np.concat(
                [
                    np.diff(
                        [
                            np.datetime64("2000-06-01 00:00:00"),
                            np.datetime64("2001-01-01 00:00:00"),
                        ]
                    ).astype("int")
                    / (366 * 60 * 60 * 24),
                    np.ones(9),
                    np.diff(
                        [
                            np.datetime64("2010-01-01 00:00:00"),
                            np.datetime64("2010-05-20 00:00:00")
                            + np.timedelta64(2, "W"),
                        ]
                    ).astype("int")
                    / (365 * 60 * 60 * 24),
                ]
            ),
            id="fortnightly_offset_both_ends",
        ),
        pytest.param(
            np.concat(  # 10 years of monthly obs with one second overlaps
                [
                    [np.datetime64("1999-12-31 23:59:59")],  # party over, out of time
                    np.arange(
                        np.datetime64("2000-01"),
                        np.datetime64("2010-02"),
                        np.timedelta64(1, "M"),
                    ),
                ]
            ),
            None,
            False,
            np.datetime64("2010-01-01 00:00:01"),
            does_not_raise(),
            None,
            np.concat(
                [[1 / (365 * 60 * 60 * 24)], np.ones(10), [1 / (365 * 60 * 60 * 24)]]
            ),
            id="second_on_each_end",
        ),
        pytest.param(
            np.concat(  # 10 years of monthly obs with one second overlaps
                [
                    [np.datetime64("1999-12-31")],  # party over, out of time
                    np.arange(
                        np.datetime64("2000"),
                        np.datetime64("2010"),
                        np.timedelta64(1, "Y"),
                    ),
                ]
            ),
            None,
            False,
            np.datetime64("2010"),
            does_not_raise(),
            None,
            np.concat([[1 / 365], np.ones(10)]),
            id="day_at_the_start",
        ),
        pytest.param(
            np.concat(  # 10 years of monthly obs with one second overlaps
                [
                    np.arange(
                        np.datetime64("2000"),
                        np.datetime64("2010"),
                        np.timedelta64(1, "Y"),
                    ),
                ]
            ),
            None,
            False,
            np.datetime64("2010-01-02"),
            does_not_raise(),
            None,
            np.concat([np.ones(10), [1 / 365]]),
            id="day_at_the_end",
        ),
    ),
)
def test_AnnualValueCalculatorMarkII_init(
    datetimes,
    as_acclim,
    growing_season,
    endpoint,
    context_manager,
    error_message,
    year_completeness,
):
    """Test failure modes and success modes for initialising AVC instances."""
    from pyrealm.phenology.fapar_limitation import AnnualValueCalculatorMarkII
    from pyrealm.pmodel.acclimation import AcclimationModel

    if as_acclim:
        datetimes = AcclimationModel(datetimes=datetimes)

    with context_manager as cmgr:
        avc = AnnualValueCalculatorMarkII(
            timing=datetimes, growing_season=growing_season, endpoint=endpoint
        )

        assert_allclose(avc.year_completeness, year_completeness)
        return

    assert str(cmgr.value) == error_message


# Global definition of 10 years of monthly data and a growing season sequence
MONTHLY: NDArray = np.arange(
    np.datetime64("2000-01"),
    np.datetime64("2010-01"),
    np.timedelta64(1, "M"),
)
GROWING_SEASON = np.tile(np.repeat([0, 1, 0], [3, 6, 3]), 10).astype(np.bool_)


@pytest.mark.parametrize(
    argnames="init, values, total_expected, total_expected_within_gs, "
    "mean_expected, mean_expected_within_gs",
    argvalues=(
        pytest.param(
            # - 10 complete years of monthly obs
            # - The test values are +0.1 in the growing season and -0.1 in the
            #   off-season to introduce explicit differences in the expected values for
            #   "within_growing_season" modes.
            # - The overall expected means including off season are slightly higher in
            #   non-leap years because there is _one more day_ in the growing season and
            #   the means are weighted by _actual duration. The two seasons are equal
            #   size in leap years.
            {
                "timing": MONTHLY,
                "endpoint": np.datetime64("2010-01"),
                "growing_season": GROWING_SEASON,
            },
            np.repeat(np.arange(1, 11), 12) + (GROWING_SEASON - 0.5) / 5,
            np.arange(1, 11) * 12,
            np.arange(1, 11) * 6 + 0.6,
            np.arange(1, 11)
            + (1 / 365 * 0.1) * np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1]),
            np.arange(1, 11) + 0.1,
            id="monthly_complete_years",
        ),
        pytest.param(
            # - Roll the monthly data forward by 6 months to get 9 complete years of
            #   monthly obs with six months either side
            # - Seasonal value adjustments as above.
            # - Non-leap year tweaks to expected means as above, plus the trailing
            #   partial year has one more day (91) in the growing season than in the off
            #   season (90), so is 1/181th larger. The leading partial year is 90/90
            {
                "timing": MONTHLY + np.timedelta64(6, "M"),
                "endpoint": np.datetime64("2010-07"),
                "growing_season": np.roll(GROWING_SEASON, 6),
            },
            np.repeat(np.arange(1, 12), [6, *[12] * 9, 6])
            + (np.roll(GROWING_SEASON, 6) - 0.5) / 5,
            np.arange(1, 12) * [6, *[12] * 9, 6],
            np.arange(1, 12) * [3, *[6] * 9, 3]
            + 0.3 * np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]),
            np.arange(1, 12)
            + 0.1
            * np.array(
                [
                    0,
                    1 / 365,
                    1 / 365,
                    1 / 365,
                    0,
                    1 / 365,
                    1 / 365,
                    1 / 365,
                    0,
                    1 / 365,
                    1 / 181,
                ]
            ),
            np.arange(1, 12) + 0.1,
            id="offset_monthly",
        ),
        pytest.param(
            # - 4 sequential non-leap years (note 1900 start) split into 10 blocks of
            #   146 days. This lengths is used because the prime factors of 365 are 5,
            #   73 and hence 146 day blocks helpfully evenly divides the spanning blocks
            #   in two for a duration weight of 0.2 years and fractional weight of 0.5.
            # - Growing season follows a simple on/off sequence. Not biologically
            #   sensible but generates good testing variation across the four years.
            {
                "timing": np.arange(
                    np.datetime64("1900-01-01"),
                    np.datetime64("1904-01-01"),
                    np.timedelta64(146, "D"),
                ),
                "growing_season": np.tile([0, 1], 5).astype(np.bool_),
            },
            np.arange(1, 11),
            np.array(
                [
                    1 + 2 + 3 * 0.5,
                    3 * 0.5 + 4 + 5,
                    6 + 7 + 8 * 0.5,
                    8 * 0.5 + 9 + 10,
                ]
            ),
            np.array(
                [
                    2,
                    4,
                    6 + 8 * 0.5,
                    8 * 0.5 + 10,
                ]
            ),
            np.array(
                [
                    (1 + 2 + 3 * 0.5) / 2.5,
                    (3 * 0.5 + 4 + 5) / 2.5,
                    (6 + 7 + 8 * 0.5) / 2.5,
                    (8 * 0.5 + 9 + 10) / 2.5,
                ]
            ),
            np.array(
                [
                    2 / 1,
                    4 / 1,
                    (6 + 8 * 0.5) / 1.5,
                    (8 * 0.5 + 10) / 1.5,
                ]
            ),
            id="spanning_years",
        ),
    ),
)
def test_AnnualValueCalculatorMarkII_get_annual(
    init,
    values,
    total_expected,
    total_expected_within_gs,
    mean_expected,
    mean_expected_within_gs,
):
    """Test the calculation of annual totals and means."""
    from pyrealm.phenology.fapar_limitation import AnnualValueCalculatorMarkII

    avc = AnnualValueCalculatorMarkII(**init)

    calculated = avc.get_annual_totals(values, within_growing_season=False)
    assert_allclose(calculated, total_expected)

    calculated = avc.get_annual_totals(values, within_growing_season=True)
    assert_allclose(calculated, total_expected_within_gs)

    calculated = avc.get_annual_means(values, within_growing_season=False)
    assert_allclose(calculated, mean_expected)

    calculated = avc.get_annual_means(values, within_growing_season=True)
    assert_allclose(calculated, mean_expected_within_gs)
