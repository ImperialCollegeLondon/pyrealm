"""Test the FaparLimitation class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="phenology_method",
    argvalues=(pytest.param("cai_zhou_method"),),
)
@pytest.mark.parametrize(
    argnames="timescale,A0_variable",
    argvalues=(
        pytest.param("hh", "annual_total_A0_smstress", id="subdaily"),
        pytest.param("ft", "annual_total_A0", id="fortnightly"),
    ),
)
def test_faparlimitation(
    site_data,
    annual_inputs,
    fapar_max_predictions,
    phenology_method,  # Used to parameterise the fapar_max_predictions fixture
    timescale,  # Also used to parameterise the annual_inputs fixture
    A0_variable,
):
    """Regression test for FaparLimitation constructor.

    This test compares the outputs of FaparLimitation for a given method against the
    predicted outputs from the reference code for each implementation.
    """

    from pyrealm.phenology.fapar_limitation import FaparLimitation

    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_inputs[A0_variable],
        annual_mean_ca=annual_inputs["annual_mean_ca_in_GS"],
        annual_mean_chi=annual_inputs["annual_mean_chi_in_GS"],
        annual_mean_vpd=annual_inputs["annual_mean_VPD_in_GS"],
        annual_total_precip=annual_inputs["annual_precip_molar"],
        annual_growing_season_length=annual_inputs["N_growing_days"],
        years=annual_inputs["year"].astype(str).astype("datetime64[Y]"),
        aridity_index=site_data["AI_from_cruts"],
    )

    assert_allclose(
        fapar_max_predictions[f"fapar_max_{timescale}"], faparlim.fapar_max, rtol=1e-6
    )
    assert_allclose(
        fapar_max_predictions[f"lai_max_{timescale}"], faparlim.lai_max, rtol=1e-6
    )

    # assert_allclose(
    #     fapar_max_predictions["m"],
    #     faparlim.lai_to_gpp_ratio_m,
    #     rtol=1e-6,
    # )


@pytest.mark.parametrize(
    argnames="phenology_method",
    argvalues=(pytest.param("cai_zhou_method"),),
)
@pytest.mark.parametrize(
    argnames="timescale,A0_variable",
    argvalues=(
        pytest.param("hh", "annual_total_A0_smstress", id="subdaily"),
        pytest.param("ft", "annual_total_A0", id="fortnightly"),
    ),
)
def test_phenology(
    site_data,
    annual_inputs,
    daily_assimilation,
    daily_lai_predictions,
    phenology_method,  # Used by pytest to parameterise the annual_inputs fixture
    timescale,
    A0_variable,
):
    """Regression test of the Phenology class on subdaily data."""
    from pyrealm.phenology.fapar_limitation import FaparLimitation, Phenology

    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_inputs[A0_variable],
        annual_mean_ca=annual_inputs["annual_mean_ca_in_GS"],
        annual_mean_chi=annual_inputs["annual_mean_chi_in_GS"],
        annual_mean_vpd=annual_inputs["annual_mean_VPD_in_GS"],
        annual_total_precip=annual_inputs["annual_precip_molar"],
        annual_growing_season_length=annual_inputs["N_growing_days"],
        years=annual_inputs["year"].astype(str).astype("datetime64[Y]"),
        aridity_index=site_data["AI_from_cruts"],
    )

    pheno = Phenology(
        daily_gpp=daily_assimilation["daily_A0"],
        datetimes=daily_assimilation["time"].astype("datetime64[D]"),
        fapar_limitation=faparlim,
    )

    # Fortnightly data is truncated by the last fortnight so need to truncate to match

    # Check the LAI time series to tolerance of data in file.
    assert_allclose(
        pheno.steady_state_LAI,
        daily_lai_predictions[f"Ls_daily_{timescale}"][: len(pheno.steady_state_LAI)],
        atol=1e-8,
    )
    assert_allclose(
        pheno.realised_LAI,
        daily_lai_predictions[f"Ls_daily_lagged_{timescale}"][
            : len(pheno.steady_state_LAI)
        ],
        atol=1e-8,
    )


@pytest.mark.parametrize(
    argnames="phenology_method",
    argvalues=(pytest.param("cai_zhou_method"),),
)
@pytest.mark.parametrize(
    argnames="timescale",
    argvalues=(
        pytest.param("hh", id="subdaily"),
        pytest.param("ft", id="fortnightly"),
    ),
)
def test_fapar_limitiation_frompmodel_to_phenology(
    site_data,
    pmodel_inputs,
    pmodel_outputs,
    daily_assimilation,
    fapar_max_predictions,
    daily_lai_predictions,
    timescale,
):
    """Regression test for from_pmodel FaparLimitation class method and Phenology.

    This combines the two tests above but uses the a PModel and the
    FaparLimitation.from_pmodel to calculate fapar max rather than using provided annual
    values.
    """

    from pyrealm.phenology.fapar_limitation import FaparLimitation, Phenology
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

    # The two timescales use different PModels and also need to specify the datetimes
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

    # Check the GPP predictions - use _gpp here to check the unpenalized values
    assert_allclose(pmodel._gpp, pmodel_outputs["gpp"], rtol=1e-7)
    assert_allclose(pmodel.optchi.ci, pmodel_outputs["ci"], rtol=1e-7)
    assert_allclose(pmodel.optchi.chi, pmodel_outputs["chi"], rtol=1e-7)

    pmodel_inputs["time"] = pmodel_inputs["time"].astype("datetime64[s]")

    faparlim = FaparLimitation.from_pmodel(
        pmodel=pmodel,
        growing_season=pmodel_inputs["growing_season"],
        datetimes=fl_datetimes,
        precip=pmodel_inputs["precip_molar"],
        aridity_index=site_data["AI_from_cruts"],
    )

    assert_allclose(
        fapar_max_predictions[f"fapar_max_{timescale}"], faparlim.fapar_max, rtol=1e-6
    )
    assert_allclose(
        fapar_max_predictions[f"lai_max_{timescale}"], faparlim.lai_max, rtol=1e-6
    )

    # The subdaily and standard pmodel._get_daily_gpp have different APIs.
    if timescale == "ft":
        days, gpp = pmodel._get_daily_gpp(datetimes=pmodel_inputs["time"])
    else:
        days, gpp = pmodel._get_daily_gpp()

    # Scale daily GPP in µmol m2 s up to daily molar assimilation.
    daily_A0 = (gpp * [60 * 60 * 24 * 1e-6]) / pmodel.core_const.k_c_molmass

    assert_allclose(daily_A0, daily_assimilation["daily_A0"], atol=1e-5)

    pheno = Phenology(daily_gpp=daily_A0, datetimes=days, fapar_limitation=faparlim)

    assert_allclose(
        pheno.steady_state_LAI,
        daily_lai_predictions[f"Ls_daily_{timescale}"][: len(pheno.steady_state_LAI)],
        atol=1e-5,
    )
    assert_allclose(
        pheno.realised_LAI,
        daily_lai_predictions[f"Ls_daily_lagged_{timescale}"][
            : len(pheno.steady_state_LAI)
        ],
        atol=1e-5,
    )
