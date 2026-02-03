"""Test the FaparLimitation class."""

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
    argnames="timescale",
    argvalues=(
        pytest.param("hh", id="subdaily"),
        pytest.param("ft", id="fortnightly"),
    ),
)
def test_pmodels(
    phenology_pmodels,
    pmodel_outputs,
    timescale,  # Parameterises phenology_pmodels and pmodel_outputs
):
    """Test fixture PModels.

    Verifies that the PModels provided by the phenology_pmodels fixture give the same
    predictions as the saved regression data.
    """

    pmodel, _, _ = phenology_pmodels

    # Check the GPP predictions
    assert_allclose(pmodel.gpp, pmodel_outputs["gpp"], rtol=1e-7)
    assert_allclose(pmodel.optchi.ci, pmodel_outputs["ci"], rtol=1e-7)
    assert_allclose(pmodel.optchi.chi, pmodel_outputs["chi"], rtol=1e-7)


@pytest.mark.parametrize(
    argnames="method_predictions_dir, method, pmodel_year",
    argvalues=(
        pytest.param("cai_zhou_method", "cai", None, id="cai"),
        pytest.param("zhu_method", "zhu", None, id="zhu"),
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
    phenology_pmodels,
    fapar_max_predictions,
    method_predictions_dir,  # Parameterises fapar_max_predictions
    method,
    pmodel_year,  # Parameterises phenology_pmodels to use all years
    timescale,  # Also parameterises phenology_pmodels, pmodel_inputs
):
    """Regression test for  FaparLimitation.from_pmodel class method."""

    from pyrealm.phenology.fapar_limitation_new import FaparLimitationNew

    pmodel, datetimes, gpp_penalty_factor = phenology_pmodels

    pmodel_inputs["time"] = pmodel_inputs["time"].astype("datetime64[s]")

    faparlim = FaparLimitationNew.from_pmodel(
        pmodel=pmodel,
        method=method,
        growing_season=pmodel_inputs["growing_season"],
        datetimes=datetimes,
        precip=pmodel_inputs["precip_molar"],
        gpp_penalty_factor=gpp_penalty_factor,
        aridity_index=site_data["AI_from_cruts"],  # Not used by zhu method.
    )

    assert_allclose(
        fapar_max_predictions[f"fapar_max_{timescale}"], faparlim.fapar_max, rtol=1e-6
    )
    assert_allclose(
        fapar_max_predictions[f"lai_max_{timescale}"], faparlim.lai_max, rtol=1e-6
    )
