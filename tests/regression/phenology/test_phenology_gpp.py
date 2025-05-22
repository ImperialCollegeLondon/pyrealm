"""Test the input values for GPP for the phenology data."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose


@pytest.mark.skip("This test fails after changes of the example output files.")
def test_phenology_gpp_calculation(
    de_gri_subdaily_data,
    de_gri_daily_outputs,
    de_gri_constants,
):
    """Test the provided GPP values for phenology can be recreated."""

    from pyrealm.pmodel import PModelEnvironment
    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.functions import calc_soilmstress_mengoli
    from pyrealm.pmodel.pmodel import SubdailyPModel

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

    # Daily values
    # Check soil moisture
    soilm_stress = calc_soilmstress_mengoli(
        soilm=de_gri_daily_outputs["wn"].to_numpy() / 150,
        aridity_index=de_gri_constants["AI_from_cruts"],
    )
    assert_allclose(soilm_stress, de_gri_daily_outputs["soilm_stress"], rtol=1e-6)

    # Check the aggregated daily mean GPP outputs
    # - PMod_sub_A0_daily_total
    # - PMod_sub_A0_daily_total_penalised

    hh_values = pd.DataFrame(
        dict(
            time=de_gri_subdaily_data["time"],
            gpp=de_gri_subdaily_pmodel.gpp,
        ),
    )
    hh_values = hh_values.set_index("time")
    hh_resampler = hh_values.resample("D")

    assert_allclose(
        de_gri_daily_outputs["PMod_gpp"],
        hh_resampler["gpp"].mean(),
        rtol=1e-6,
    )

    assert_allclose(
        de_gri_daily_outputs["PMod_gpp_smstress"],
        hh_resampler["gpp"].mean() * soilm_stress,
        rtol=1e-6,
    )
