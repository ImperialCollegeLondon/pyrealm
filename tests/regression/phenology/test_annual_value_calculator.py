"""Testing the annual value calculator."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose


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
    " context_manager, error_message",
    argvalues=(
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
            None,
            pytest.raises(ValueError),
            "The timings argument must be an AcclimationModel "
            "or an array of datetime64 values",
            id="timings_not_acclim_or_datetimes",
        ),
        pytest.param(
            np.concat(
                [
                    [np.datetime64("2000-01-01 00:00:00")],
                    np.arange(
                        np.datetime64("2000-01-01"),
                        np.datetime64("2010-01-01"),
                        np.timedelta64(30, "m"),
                    ),
                ]
            ),
            np.ones(
                (365 * 5 + 366 * 3) * 48 + 1, dtype=np.bool_
            ),  # 10 years of 30 min obs, with 3 leap years with a duplicate at start
            False,
            None,
            pytest.raises(ValueError),
            "The timing values are not strictly increasing",
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
            id="bad_growing_season_dtype",
        ),
        # pytest.param(
        #     np.arange(
        #         np.datetime64("2000-01-31"),
        #         np.datetime64("2009-11-30"),
        #         np.timedelta64(30, "m"),
        #     ),
        #     np.ones(
        #         (365 * 7 + 366 * 3) * 48, dtype=np.bool_
        #     ),  # 10 years of 30 min obs, with 3 leap years
        #     True,
        #     pytest.raises(ValueError),
        #     "Data timings do not cover complete years to within tolerance",
        #     id="acclim_model_incomplete_years",
        # ),
        # pytest.param(
        #     np.arange(
        #         np.datetime64("2000-01-01"),
        #         np.datetime64("2010-01-01"),
        #         np.timedelta64(1, "D"),
        #     ),
        #     np.ones(
        #         (365 * 7 + 366 * 3), dtype=np.bool_
        #     ),  # 10 years of daily obs, with 3 leap years
        #     False,
        #     does_not_raise(),
        #     None,
        #     id="datetimes_daily_good",
        # ),
        # pytest.param(
        #     np.arange(
        #         np.datetime64("2000-01-01"),
        #         np.datetime64("2010-01-01"),
        #         np.timedelta64(1, "W"),
        #     ),
        #     np.ones(522, dtype=np.bool_),  # 10 years of weekly obs
        #     False,
        #     does_not_raise(),
        #     None,
        #     id="datetimes_weekly_good",
        # ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.timedelta64(2, "W"),
            ),
            np.ones(261, dtype=np.bool_),  # 10 years of fortnightly obs
            False,
            None,
            does_not_raise(),
            None,
            id="fortnightly",
        ),
        pytest.param(
            np.arange(
                np.datetime64("2000-01"),
                np.datetime64("2010-01"),
                np.timedelta64(1, "M"),
            ),
            np.ones(120, dtype=np.bool_),  # 10 years of monthly obs
            False,
            None,
            does_not_raise(),
            None,
            id="monthly",
        ),
        # pytest.param(
        #     np.arange(
        #         np.datetime64("2000-01-08"),
        #         np.datetime64("2010-01-01"),
        #         np.timedelta64(1, "W"),
        #     ),
        #     np.ones(522, dtype=np.bool_),  # 10 years of weekly obs
        #     False,
        #     does_not_raise(),
        #     None,
        #     id="datetimes_weekly_short_outside_tolerance",
        # ),
        #         pytest.param(
        #     np.arange(
        #         np.datetime64("2000-01-01"),
        #         np.datetime64("2010-01-01"),
        #         np.timedelta64(30, "m"),
        #     ),
        #     np.ones(
        #         (365 * 7 + 366 * 3) * 48, dtype=np.bool_
        #     ),  # 10 years of 30 min obs, with 3 leap years
        #     True,
        #     does_not_raise(),
        #     None,
        #     id="acclim_model_good",
        # ),
        # pytest.param(
        #     np.arange(
        #         np.datetime64("2000-01-01"),
        #         np.datetime64("2010-01-01"),
        #         np.timedelta64(30, "m"),
        #     ),
        #     np.ones(
        #         (365 * 7 + 366 * 3) * 48, dtype=np.bool_
        #     ),  # 10 years of 30 min obs, with 3 leap years
        #     False,
        #     does_not_raise(),
        #     None,
        #     id="datetimes_half_hourly_good",
        # ),
    ),
)
def test_AnnualValueCalculatorMarkII_init(
    datetimes, as_acclim, growing_season, endpoint, context_manager, error_message
):
    """Test failure modes and success modes for initialising AVC instances."""
    from pyrealm.phenology.fapar_limitation import AnnualValueCalculatorMarkII
    from pyrealm.pmodel.acclimation import AcclimationModel

    if as_acclim:
        datetimes = AcclimationModel(datetimes=datetimes)

    with context_manager as cmgr:
        _ = AnnualValueCalculatorMarkII(
            timing=datetimes, growing_season=growing_season, endpoint=endpoint
        )
        return

    assert str(cmgr.value) == error_message
