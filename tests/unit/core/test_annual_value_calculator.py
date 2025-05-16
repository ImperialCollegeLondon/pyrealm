"""Testing the annual value calculator."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from numpy.typing import NDArray


@pytest.mark.parametrize(
    argnames="datetimes, growing_season,  as_acclim, endpoint,"
    " context_manager, error_message, expected",
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
            np.arange(  # 10 years of fortnightly obs with default growing season
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(2, "W"),
            ),
            None,
            False,
            None,
            does_not_raise(),
            None,
            (
                np.concat([np.ones(10), [1 / 365]]),
                np.repeat([366, 365, 366, 365, 366, 365], [1, 3, 1, 3, 1, 2]),
                np.repeat([366, 365, 366, 365, 366, 365, 1], [1, 3, 1, 3, 1, 1, 1]),
                np.arange(
                    np.datetime64("2000"), np.datetime64("2012"), np.timedelta64(1, "Y")
                ),
            ),
            id="fortnightly",
        ),
        pytest.param(
            np.arange(  # 10 years of monthly obs with default growing season
                np.datetime64("2000-01"),
                np.datetime64("2010-01"),
                np.timedelta64(1, "M"),
            ),
            None,
            False,
            np.datetime64("2010-01"),
            does_not_raise(),
            None,
            (
                np.ones(10),
                np.repeat([366, 365, 366, 365, 366, 365], [1, 3, 1, 3, 1, 1]),
                np.repeat([366, 365, 366, 365, 366, 365], [1, 3, 1, 3, 1, 1]),
                np.arange(
                    np.datetime64("2000"), np.datetime64("2011"), np.timedelta64(1, "Y")
                ),
            ),
            id="monthly",
        ),
        pytest.param(
            # 10 years of fortnightly obs but offset by half a year with default
            # growing season
            np.arange(
                np.datetime64("2000-06-01"),
                np.datetime64("2010-06-01"),
                np.timedelta64(2, "W"),
            ),
            None,
            False,
            None,
            does_not_raise(),
            None,
            (
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
                np.repeat([366, 365, 366, 365, 366, 365], [1, 3, 1, 3, 1, 2]),
                np.repeat([214, 365, 366, 365, 366, 365, 153], [1, 3, 1, 3, 1, 1, 1]),
                np.arange(
                    np.datetime64("2000"), np.datetime64("2012"), np.timedelta64(1, "Y")
                ),
            ),
            id="fortnightly_offset_both_ends",
        ),
        pytest.param(
            # 10 years of monthly obs with one second overlaps with default
            # growing season
            np.concat(
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
            (
                np.concat(
                    [
                        [1 / (365 * 60 * 60 * 24)],
                        np.ones(10),
                        [1 / (365 * 60 * 60 * 24)],
                    ]
                ),
                np.repeat([365, 366, 365, 366, 365, 366, 365], [1, 1, 3, 1, 3, 1, 2]),
                np.repeat(
                    [1 / 86400, 366, 365, 366, 365, 366, 365, 1 / 86400],
                    [1, 1, 3, 1, 3, 1, 1, 1],
                ),
                np.arange(
                    np.datetime64("1999"), np.datetime64("2012"), np.timedelta64(1, "Y")
                ),
            ),
            id="second_on_each_end",
        ),
        pytest.param(
            # 10 years of yearly data with one day at start with default growing season
            np.concat(
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
            (
                np.concat([[1 / 365], np.ones(10)]),
                np.repeat([365, 366, 365, 366, 365, 366, 365], [1, 1, 3, 1, 3, 1, 1]),
                np.repeat([1, 366, 365, 366, 365, 366, 365], [1, 1, 3, 1, 3, 1, 1]),
                np.arange(
                    np.datetime64("1999"), np.datetime64("2011"), np.timedelta64(1, "Y")
                ),
            ),
            id="day_at_the_start",
        ),
        pytest.param(
            # 10 years of yearly data with one day at end with default growing season
            np.concat(
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
            (
                np.concat([np.ones(10), [1 / 365]]),
                np.repeat([366, 365, 366, 365, 366, 365], [1, 3, 1, 3, 1, 2]),
                np.repeat([366, 365, 366, 365, 366, 365, 1], [1, 3, 1, 3, 1, 1, 1]),
                np.arange(
                    np.datetime64("2000"), np.datetime64("2012"), np.timedelta64(1, "Y")
                ),
            ),
            id="day_at_the_end",
        ),
        pytest.param(
            np.arange(  # 10 years of monthly obs with summer growing season
                np.datetime64("2000-01"),
                np.datetime64("2010-01"),
                np.timedelta64(1, "M"),
            ),
            np.tile(np.repeat([False, True, False], [3, 6, 3]), 10),
            False,
            np.datetime64("2010-01"),
            does_not_raise(),
            None,
            (
                np.ones(10),
                np.repeat([366, 365, 366, 365, 366, 365], [1, 3, 1, 3, 1, 1]),
                np.repeat(183, 10),  # April- September 30 + 31 + 30 + 31 + 31 + 30
                np.arange(
                    np.datetime64("2000"), np.datetime64("2011"), np.timedelta64(1, "Y")
                ),
            ),
            id="monthly_summer_growing_season",
        ),
    ),
)
def test_AnnualValueCalculator_init(
    datetimes,
    as_acclim,
    growing_season,
    endpoint,
    context_manager,
    error_message,
    expected,
):
    """Test failure modes and success modes for initialising AVC instances.

    The weightings are tested by the tests of the get_annual_totals and get_annual_means
    methods. This tests the year completeness and the number of days and growing days.
    """
    from pyrealm.core.time_series import AnnualValueCalculator
    from pyrealm.pmodel.acclimation import AcclimationModel

    if as_acclim:
        datetimes = AcclimationModel(datetimes=datetimes)

    with context_manager as cmgr:
        avc = AnnualValueCalculator(
            timing=datetimes, growing_season=growing_season, endpoint=endpoint
        )

        year_completeness, year_n_days, year_n_growing_days, years = expected

        assert_allclose(avc.year_completeness, year_completeness)
        assert_allclose(avc.year_n_days, year_n_days)
        assert_allclose(avc.year_n_growing_days, year_n_growing_days)
        assert_equal(avc.years, years)
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
            #   146 days. This length is used because the prime factors of 365 are 5,
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
        pytest.param(
            # - 10 complete years of monthly obs with missing data. One equal sized
            #   month is missing from each of the growing season (May) and off season
            #   (Jan), so the -0.1 and +0.1 differences and weightings cancel out in the
            #   whole year sums but don't in the growing season.
            # - The calculation of means is awkward - in non-leap years, there is one
            #   fewer day in the valued parts of the data, so an extra 0.1 across 303
            #   days with values.
            {
                "timing": MONTHLY,
                "endpoint": np.datetime64("2010-01"),
                "growing_season": GROWING_SEASON,
            },
            (np.repeat(np.arange(1, 11), 12) + (GROWING_SEASON - 0.5) / 5)
            * np.tile(np.repeat([np.nan, 1, np.nan, 1], [1, 3, 1, 7]), 10),
            np.arange(1, 11) * 10,
            np.arange(1, 11) * 5 + 5 * 0.1,
            np.arange(1, 11) + (0.1 / 303) * np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1]),
            np.arange(1, 11) + 0.1,
            id="monthly_complete_years_with_nan",
        ),
    ),
)
def test_AnnualValueCalculator_get_annual(
    init,
    values,
    total_expected,
    total_expected_within_gs,
    mean_expected,
    mean_expected_within_gs,
):
    """Test the calculation of annual totals and means."""
    from pyrealm.core.time_series import AnnualValueCalculator

    avc = AnnualValueCalculator(**init)

    calculated = avc.get_annual_totals(values, within_growing_season=False)
    assert_allclose(calculated, total_expected)

    calculated = avc.get_annual_totals(values, within_growing_season=True)
    assert_allclose(calculated, total_expected_within_gs)

    calculated = avc.get_annual_means(values, within_growing_season=False)
    assert_allclose(calculated, mean_expected)

    calculated = avc.get_annual_means(values, within_growing_season=True)
    assert_allclose(calculated, mean_expected_within_gs)
