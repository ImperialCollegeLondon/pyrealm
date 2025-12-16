"""Test the input values for GPP for the phenology data."""

import numpy as np
from numpy.testing import assert_allclose


def test_phenology_gpp_calculation(
    de_gri_splash_data,
    de_gri_subdaily_data,
    de_gri_daily_outputs,
    de_gri_fortnightly_data,
    de_gri_fortnightly_daily_outputs,
    de_gri_constants,
):
    """Test the provided GPP values for phenology can be recreated."""

    from pyrealm.pmodel import PModelEnvironment
    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.functions import calc_soilmstress_mengoli
    from pyrealm.pmodel.pmodel import PModel, SubdailyPModel

    # Calculate the PModel photosynthetic environment
    env = PModelEnvironment(
        tc=de_gri_subdaily_data["tc"].to_numpy(),
        vpd=de_gri_subdaily_data["vpd"].to_numpy(),
        co2=de_gri_subdaily_data["co2"].to_numpy(),
        patm=de_gri_subdaily_data["patm"].to_numpy(),
        fapar=np.ones(de_gri_subdaily_data.shape[0]),
        ppfd=de_gri_subdaily_data["ppfd"].to_numpy(),
    )

    # Set up the datetimes of the observations and set the acclimation window
    acclim_model = AcclimationModel(datetimes=de_gri_subdaily_data["time"].to_numpy())
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

    assert_allclose(
        de_gri_subdaily_pmodel.gpp, de_gri_subdaily_data["PMod_gpp"], rtol=1e-6
    )

    assert_allclose(
        de_gri_subdaily_pmodel.optchi.ci, de_gri_subdaily_data["PMod_ci"], rtol=1e-6
    )

    # Check soil moisture stress calculation, extract daily values for soil moisture
    soil_moisture = (
        de_gri_splash_data["wn"]
        .sel(
            time=slice(
                de_gri_subdaily_data["time"].min(), de_gri_subdaily_data["time"].max()
            )
        )
        .to_numpy()
    )

    soilm_stress = calc_soilmstress_mengoli(
        soilm=soil_moisture / 150,
        aridity_index=de_gri_constants["AI_from_cruts"],
    )
    # Check the half hourly values in the subdaily data is correct
    assert_allclose(
        np.repeat(soilm_stress, 48), de_gri_subdaily_data["soilm_stress"], rtol=1e-6
    )

    # Check the aggregated daily mean GPP outputs
    # - PMod_sub_A0_daily_total
    # - PMod_sub_A0_daily_total_penalised

    daily_gpp = de_gri_subdaily_pmodel._get_daily_gpp()

    assert_allclose(
        de_gri_daily_outputs["PMod_gpp_smstress"],
        daily_gpp * soilm_stress,
        rtol=1e-6,
    )

    fortnightly_env = PModelEnvironment(
        tc=de_gri_fortnightly_data["tc_mean"].to_numpy(),
        vpd=de_gri_fortnightly_data["vpd_mean"].to_numpy(),
        patm=de_gri_fortnightly_data["patm_mean"].to_numpy(),
        co2=de_gri_fortnightly_data["co2_mean"].to_numpy(),
        ppfd=de_gri_fortnightly_data["ppfd_mean"].to_numpy(),
        fapar=np.ones_like(de_gri_fortnightly_data["tc_mean"]),
    )
    fortnightly_datetimes = de_gri_fortnightly_data["time"]
    de_gri_pmodel = PModel(env=fortnightly_env)
    pmodel_gpp_from_fortnightly = de_gri_pmodel._get_daily_gpp(
        datetimes=fortnightly_datetimes
    )

    assert_allclose(
        de_gri_fortnightly_daily_outputs["daily_gpp"],
        pmodel_gpp_from_fortnightly[:-1],
        rtol=1e-6,
    )
