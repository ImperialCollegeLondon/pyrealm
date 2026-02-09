"""Test the FaparLimitation class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="timescale,assim_var",
    argvalues=(
        pytest.param("ft", "annual_total_A0", id="fortnightly"),
        pytest.param("hh", "annual_total_A0_smstress", id="subdaily"),
    ),
)
@pytest.mark.parametrize(
    argnames="method_predictions_dir, method",
    argvalues=(
        pytest.param("cai_zhou_method", "cai", id="cai"),
        pytest.param("zhu_method", "zhu", id="zhu"),
    ),
)
def test_faparlimitation(
    site_data,
    annual_inputs,
    fapar_max_predictions,
    timescale,  # parameterises annual_inputs fixture
    assim_var,
    method_predictions_dir,  # parameterises fapar_max_predictions fixture
    method,
):
    """Regression test for FaparLimitation constructor with fortnightly data."""

    from pyrealm.phenology.fapar_limitation_new import FaparLimitationNew

    faparlim = FaparLimitationNew(
        annual_total_potential_gpp=annual_inputs[assim_var],
        annual_mean_ca=annual_inputs["annual_mean_ca_in_GS"],
        annual_mean_chi=annual_inputs["annual_mean_chi_in_GS"],
        annual_mean_vpd=annual_inputs["annual_mean_VPD_in_GS"],
        annual_total_precip=annual_inputs["annual_precip_molar"],
        annual_growing_season_length=annual_inputs["N_growing_days"],
        years=annual_inputs["year"].astype(str).astype("datetime64[Y]"),
        method=method,
        aridity_index=site_data["AI_from_cruts"],  # Not used by zhu method.
    )

    assert_allclose(
        fapar_max_predictions[f"fapar_max_{timescale}"],
        faparlim.fapar_max,
        rtol=1e-6,
    )
    assert_allclose(
        fapar_max_predictions[f"lai_max_{timescale}"],
        faparlim.lai_max,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    argnames="method_predictions_dir, method",
    argvalues=(
        pytest.param("cai_zhou_method", "cai", id="cai"),
        pytest.param("zhu_method", "zhu", id="zhu"),
    ),
)
@pytest.mark.parametrize(
    argnames="timescale",
    argvalues=(
        pytest.param("hh", id="subdaily"),
        pytest.param("ft", id="fortnightly"),
    ),
)
def test_fapar_limitation_frompmodel(
    site_data,
    pmodel_inputs,
    pmodel_outputs,
    daily_assimilation,
    fapar_max_predictions,
    daily_lai_predictions,
    method_predictions_dir,
    method,
    timescale,
):
    """Regression test for  FaparLimitation.from_pmodel class method."""

    from pyrealm.phenology.fapar_limitation_new import FaparLimitationNew
    from pyrealm.pmodel import (
        AcclimationModel,
        PModel,
        PModelEnvironment,
        SubdailyPModel,
    )

    env = PModelEnvironment(
        tc=pmodel_inputs["tc"],
        vpd=pmodel_inputs["vpd"],
        co2=pmodel_inputs["co2"],
        patm=pmodel_inputs["patm"],
        fapar=pmodel_inputs["fapar"],
        ppfd=pmodel_inputs["ppfd"],
    )

    pmodel_inputs["time"] = pmodel_inputs["time"].astype("datetime64[s]")

    # The two timescales use different PModels and datetime sequences. Also need to
    # apply the soil moisture stress factor to the subdaily timescale.
    # and gpp penalty factors in FaparLimitation differently
    if timescale == "ft":
        # Fit PModel
        pmodel = PModel(
            env=env,
            reference_kphio=1 / 8,
            method_kphio="temperature",
        )
        # Define datetimes of observations - no GPP penalty
        fl_datetimes = pmodel_inputs["time"]

    else:
        # Set up the datetimes of the observations and set the acclimation window
        acclim = AcclimationModel(
            datetimes=pmodel_inputs["time"],
            alpha=1 / 15,
        )
        acclim.set_window(
            window_center=np.timedelta64(12, "h"),
            half_width=np.timedelta64(30, "m"),
        )

        # Fit the subdaily PModel
        pmodel = SubdailyPModel(
            env=env,
            acclim_model=acclim,
            reference_kphio=1 / 8,
            method_kphio="temperature",
        )

        # FaparLimitation uses the datetimes from pmodel.acclim_model and uses a soil
        # moisture stress penalty
        fl_datetimes = None
        pmodel.apply_gpp_penalty_factor(pmodel_inputs["soilm_stress"])

    # Check the GPP predictions - using _gpp to test the raw GPP rather than with any
    # penalty factor applied.
    assert_allclose(pmodel._gpp, pmodel_outputs["gpp"], rtol=1e-7)
    assert_allclose(pmodel.optchi.ci, pmodel_outputs["ci"], rtol=1e-7)
    assert_allclose(pmodel.optchi.chi, pmodel_outputs["chi"], rtol=1e-7)

    pmodel_inputs["time"] = pmodel_inputs["time"].astype("datetime64[s]")

    faparlim = FaparLimitationNew.from_pmodel(
        pmodel=pmodel,
        method=method,
        growing_season=pmodel_inputs["growing_season"],
        datetimes=fl_datetimes,
        precip=pmodel_inputs["precip_molar"],
        aridity_index=site_data["AI_from_cruts"],  # Not used by zhu method.
    )

    assert_allclose(
        fapar_max_predictions[f"fapar_max_{timescale}"], faparlim.fapar_max, rtol=1e-6
    )
    assert_allclose(
        fapar_max_predictions[f"lai_max_{timescale}"], faparlim.lai_max, rtol=1e-6
    )
