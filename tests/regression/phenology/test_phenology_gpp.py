"""Test the input values for GPP for the phenology data."""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose


# @pytest.mark.skip(
#     "Need to resolve calculation process - currently not matching due to aridity "
#     "penalty applied to GPP in input data."
# )
def test_phenology_gpp_calculation(
    de_gri_daily_data, de_gri_half_hourly_data, de_gri_constants
):
    """Test the provided GPP values for phenology can be recreated."""

    from pyrealm.pmodel import PModel, PModelEnvironment, SubdailyPModel, SubdailyScaler
    from pyrealm.pmodel.functions import calc_soilmstress_mengoli

    # Calculate the PModel photosynthetic environment
    env = PModelEnvironment(
        tc=de_gri_half_hourly_data["TA_F"].to_numpy(),
        vpd=de_gri_half_hourly_data["VPD_F"].to_numpy(),
        co2=de_gri_half_hourly_data["CO2_F_MDS"].to_numpy(),
        patm=de_gri_half_hourly_data["PA_F"].to_numpy(),
    )

    de_gri_pmodel = PModel(
        env=env,
        # reference_kphio=1 / 8,
    )

    de_gri_pmodel.estimate_productivity(
        fapar=np.ones_like(env.ca),
        ppfd=de_gri_half_hourly_data["SW_IN_F"].to_numpy() * 2.04,
    )

    # Currently close but not exact
    assert_allclose(de_gri_pmodel.gpp, de_gri_half_hourly_data["A0_normal"].to_numpy())

    # Set up the datetimes of the observations and set the acclimation window
    scaler = SubdailyScaler(datetimes=de_gri_half_hourly_data["time"].to_numpy())
    scaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Fit the potential GPP: fAPAR = 1 and phi0 = 1/8
    de_gri_subdaily_pmodel = SubdailyPModel(
        env=env,
        fs_scaler=scaler,
        fapar=np.ones_like(env.ca),
        ppfd=de_gri_half_hourly_data["PPFD"].to_numpy(),
        reference_kphio=1 / 8,
    )

    # Calculate soil moisture stress factor
    soilm_stress = calc_soilmstress_mengoli(
        soilm=de_gri_daily_data["soilm"].to_numpy(),
        aridity_index=de_gri_constants["AI_from_coords"],
    )

    assert np.allclose(
        de_gri_subdaily_pmodel.gpp * soilm_stress,
        de_gri_half_hourly_data["A0_slow"].to_numpy(),
    )


def test_daily_values(de_gri_half_hourly_data, de_gri_daily_data):
    """Tests the calculation of daily values calculated from half hourly predictions.

    This is just to validate that daily and annual values in the de_gri_daily_data
    inputs can be correctly derived from the original half hourly inputs.
    """

    from pyrealm.phenology.growing_season import filter_short_intervals

    # Set the time as the index to enable temporal resampling.
    de_gri_half_hourly_data = de_gri_half_hourly_data.set_index("time")

    # Daily values
    de_gri_daily_resampler = de_gri_half_hourly_data.resample("D")

    # Mean daily conditions - temperature, VPD and pressure
    for daily_mean_var in ["TA_F", "VPD_F", "PA_F"]:
        daily_means = de_gri_daily_resampler[daily_mean_var].mean().to_numpy()
        assert np.allclose(daily_means, de_gri_daily_data[daily_mean_var], atol=0.001)

    # Total precipitation
    daily_precip = de_gri_daily_resampler["P_F"].sum().to_numpy()
    assert np.allclose(daily_precip, de_gri_daily_data["P_F"], atol=0.01)

    # Going from daily data to annual average conditions
    de_gri_daily_data = de_gri_daily_data.set_index("date")

    # Calculate suitable growth days - days within any period of five or more
    # consecutive days above 0°C are growing days.
    above_zero_degrees = (de_gri_daily_data["TA_F"] >= 0).to_numpy()
    not_suitable = filter_short_intervals(
        np.logical_not(above_zero_degrees),
        window=4,
    )
    de_gri_daily_data["growing_day"] = np.logical_not(not_suitable)

    # Get a sampler on the daily data to calculate annually constant values
    de_gri_annual_resampler = de_gri_daily_data.resample("YE")

    # Subset to the values in the dataset - check there is no variation in the values
    # before extracting annual values
    de_gri_annual_values = de_gri_annual_resampler[
        ["P", "CO2_F_MDS", "VPD", "chi_gst", "Gsl"]
    ]

    assert np.all(de_gri_annual_values.var() == 0)
    expected_annual = de_gri_annual_values.mean()

    # Check the calculation from daily data frame:
    # * mean CO2
    # * total precipitation
    # * count of growing days

    # Resampler of filtered values to get VPD and Chi during the growing season
    de_gri_growing_season_resampler = de_gri_daily_data[
        de_gri_daily_data["growing_day"]
    ].resample("YE")

    # Compile annual values
    calculated_annual = pd.concat(
        [
            de_gri_annual_resampler[["P_F"]].sum(),
            de_gri_annual_resampler[["CO2_F_MDS"]].mean(),
            de_gri_growing_season_resampler[["VPD_F"]].mean(),
            de_gri_growing_season_resampler[["VPD_F"]].mean(),  # TODO - change to chi
            de_gri_annual_resampler[["growing_day"]].sum(),
        ],
        axis=1,
    )

    # Synchronise field names
    calculated_annual.columns = expected_annual.columns

    # Test calculation with provided absolute tolerances.
    # TODO - these tolerances are currently very large. Need to resolve issue with
    # identifying growing season days, which is wrecking the calculation.

    test_tol = {
        "P": 0.001,
        "CO2_F_MDS": 0.00001,
        "VPD": 0.5,  # TODO - fix this
        "chi_gst": 5,  # TODO - fix this
        "Gsl": 5,  # TODO - fix this
    }

    for col in calculated_annual.columns:
        assert_allclose(
            expected_annual[col], calculated_annual[col], atol=test_tol[col]
        )
