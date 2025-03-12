"""Test the input values for GPP for the phenology data."""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose


def test_phenology_gpp_calculation(
    de_gri_half_hourly_inputs,
    de_gri_splash_inputs,
    de_gri_hh_outputs,
    de_gri_daily_outputs,
    de_gri_constants,
):
    """Test the provided GPP values for phenology can be recreated."""

    from pyrealm.pmodel import PModelEnvironment
    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.functions import calc_soilmstress_mengoli
    from pyrealm.pmodel.pmodel import PModel, SubdailyPModel

    # Calculate the PModel photosynthetic environment
    env = PModelEnvironment(
        tc=de_gri_half_hourly_inputs["TA_F"].to_numpy(),
        vpd=de_gri_half_hourly_inputs["VPD_F"].to_numpy(),
        co2=de_gri_half_hourly_inputs["CO2_F_MDS"].to_numpy(),
        patm=de_gri_half_hourly_inputs["PA_F"].to_numpy(),
        fapar=np.ones(de_gri_half_hourly_inputs.shape[0]),
        ppfd=de_gri_half_hourly_inputs["PPFD"].to_numpy(),
    )

    # Calculate soil moisture stress factor
    # - double check the aridity index across 20 years matches the inputs
    aridity_index = float(
        de_gri_splash_inputs["pet"].sum() / de_gri_splash_inputs["pre"].sum()
    )

    assert_allclose(de_gri_constants["AI_from_cruts"], aridity_index)

    # reduce to match time series
    de_gri_splash_inputs = de_gri_splash_inputs.sel(
        time=slice("2004-01-01", "2014-12-31")
    )

    soilm_stress = calc_soilmstress_mengoli(
        soilm=de_gri_splash_inputs["wn"].to_numpy() / 150,
        aridity_index=de_gri_constants["AI_from_cruts"],
    )

    assert_allclose(soilm_stress, de_gri_daily_outputs["soilm_stress"], rtol=1e-6)

    # Standard PModel - not used in further calculations but should agree
    de_gri_pmodel = PModel(
        env=env,
        reference_kphio=1 / 8,
    )
    assert_allclose(de_gri_pmodel.gpp, de_gri_hh_outputs["PMod_A0"], rtol=1e-6)

    # Set up the datetimes of the observations and set the acclimation window
    acclim_model = AcclimationModel(
        datetimes=de_gri_half_hourly_inputs["time"].to_numpy()
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

    assert_allclose(
        de_gri_subdaily_pmodel.gpp, de_gri_hh_outputs["PMod_sub_A0"], rtol=1e-6
    )

    assert_allclose(
        de_gri_subdaily_pmodel.optchi.ci, de_gri_hh_outputs["PMod_sub_ci"], rtol=1e-6
    )

    # Double check the aggregated daily GPP outputs
    # - PMod_A0_daily_total
    # - PMod_sub_A0_daily_total
    # - PMod_sub_A0_daily_total_penalised

    hh_values = pd.DataFrame(
        dict(
            time=de_gri_half_hourly_inputs["time"],
            PMod_A0=de_gri_pmodel.gpp,
            PMod_sub_A0=de_gri_subdaily_pmodel.gpp,
        ),
    )
    hh_values = hh_values.set_index("time")
    hh_resampler = hh_values.resample("D")

    assert_allclose(
        de_gri_daily_outputs["PMod_A0_daily_total"],
        hh_resampler["PMod_A0"].mean() * ((60 * 60 * 24) / 1e6),
        rtol=1e-6,
    )

    assert_allclose(
        de_gri_daily_outputs["PMod_sub_A0_daily_total"],
        hh_resampler["PMod_sub_A0"].mean() * ((60 * 60 * 24) / 1e6),
        rtol=1e-6,
    )

    assert_allclose(
        de_gri_daily_outputs["PMod_sub_A0_daily_total_penalised"],
        hh_resampler["PMod_sub_A0"].mean() * ((60 * 60 * 24) / 1e6) * soilm_stress,
        rtol=1e-6,
    )
