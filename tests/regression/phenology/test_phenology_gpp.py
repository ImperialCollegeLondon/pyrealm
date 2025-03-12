"""Test the input values for GPP for the phenology data."""

import numpy as np
from numpy.testing import assert_allclose


# @pytest.mark.skip(
#     "Need to resolve calculation process - currently not matching due to aridity "
#     "penalty applied to GPP in input data."
# )
def test_phenology_gpp_calculation(
    de_gri_daily_data, de_gri_half_hourly_data, de_gri_constants
):
    """Test the provided GPP values for phenology can be recreated."""

    from pyrealm.pmodel import PModelEnvironment
    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.functions import calc_soilmstress_mengoli
    from pyrealm.pmodel.pmodel import PModel, SubdailyPModel

    # Calculate the PModel photosynthetic environment
    env = PModelEnvironment(
        tc=de_gri_half_hourly_data["TA_F"].to_numpy(),
        vpd=de_gri_half_hourly_data["VPD_F"].to_numpy(),
        co2=de_gri_half_hourly_data["CO2_F_MDS"].to_numpy(),
        patm=de_gri_half_hourly_data["PA_F"].to_numpy(),
        fapar=np.ones(de_gri_half_hourly_data.shape[0]),
        # ppfd=de_gri_half_hourly_data["PPFD"].to_numpy(),
        ppfd=de_gri_half_hourly_data["SW_IN_F_MDS"].to_numpy() * 2.04,
    )

    de_gri_pmodel = PModel(
        env=env,
        reference_kphio=1 / 8,
    )

    de_gri_pmodel.estimate_productivity(
        fapar=np.ones_like(env.ca),
        ppfd=de_gri_half_hourly_data["SW_IN_F"].to_numpy() * 2.04,
    )

    # Load alternative soil moisture data
    # datapath = (
    #     resources.files("pyrealm_build_data.phenology") / "soil_moisture_data.nc"
    # )
    # soil_m_data = xr.load_dataset(datapath)

    # Calculate soil moisture stress factor
    soilm_stress = calc_soilmstress_mengoli(
        soilm=de_gri_daily_data["soilm"].to_numpy(),
        aridity_index=de_gri_constants["AI_from_coords"],
    )

    # Currently close but not exact
    assert_allclose(
        de_gri_pmodel.gpp * soilm_stress,
        de_gri_half_hourly_data["A0_normal"].to_numpy(),
    )

    # Set up the datetimes of the observations and set the acclimation window
    acclim_model = AcclimationModel(
        datetimes=de_gri_half_hourly_data["time"].to_numpy()
    )
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Fit the potential GPP: fAPAR = 1 and phi0 = 1/8
    de_gri_subdaily_pmodel = SubdailyPModel(
        env=env,
        acclim_model=acclim_model,
        reference_kphio=1 / 8,
    )

    de_gri_pmodel = PModel(
        env=env,
        reference_kphio=1 / 8,
    )

    # Currently close but not exact
    assert_allclose(de_gri_pmodel.gpp, de_gri_half_hourly_data["A0_normal"].to_numpy())
    assert_allclose(
        de_gri_subdaily_pmodel.gpp, de_gri_half_hourly_data["A0_slow"].to_numpy()
    )


def test_daily_values(de_gri_half_hourly_data, de_gri_daily_data):
    """Tests the calculation of daily values calculated from half hourly predictions.

    This is just to validate that daily and annual values in the de_gri_daily_data
    inputs can be correctly derived from the original half hourly inputs.

    The tolerances on this are very coarse because the source file has rounding issues.
    """

    from pyrealm.phenology.growing_season import filter_short_intervals

    # Set the time as the index to enable temporal resampling.
    de_gri_half_hourly_data = de_gri_half_hourly_data.set_index("time")

    # Daily values
    de_gri_daily_resampler = de_gri_half_hourly_data.resample("D")

    # Mean daily conditions - temperature, VPD and pressure
    for daily_mean_var, tols in [
        ("TA_F", {"atol": 0.001}),  # Temperatures rounded to 3 dp
        ("VPD_F", {"atol": 0.001}),  # VPD rounded to 3 dp
        ("PA_F", {"atol": 1}),  # Atmospheric pressure rounded to 0 dp
    ]:
        daily_means = de_gri_daily_resampler[daily_mean_var].mean().to_numpy()
        assert_allclose(daily_means, de_gri_daily_data[daily_mean_var], **tols)

    # Total precipitation
    daily_precip = de_gri_daily_resampler["P_F"].sum().to_numpy()
    assert_allclose(daily_precip, de_gri_daily_data["P_F"], atol=0.01)

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
    # de_gri_annual_resampler = de_gri_daily_data.resample("YE")

    assert_allclose(
        de_gri_daily_data["CO2_F_MDS"],
        de_gri_daily_data["CO2_F_MDS_validation"],
        atol=1,
    )

    assert_allclose(de_gri_daily_data["P"], de_gri_daily_data["P_F_validation"], atol=1)
