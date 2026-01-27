"""Test the FaparLimitation class."""

import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="timescale,timescale_abbr,assim_var",
    argvalues=(
        pytest.param("fortnightly", "ft", "annual_total_A0", id="fortnightly"),
        pytest.param("subdaily", "hh", "annual_total_A0_smstress", id="subdaily"),
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
    data_fapar_limitation,
    timescale,  # parameterises data_fapar_limitation fixture
    timescale_abbr,
    assim_var,
    method_predictions_dir,  # parameterises data_fapar_limitation fixture
    method,
):
    """Regression test for FaparLimitation constructor with fortnightly data."""

    from pyrealm.phenology.fapar_limitation_new import FaparLimitationNew

    faparlim = FaparLimitationNew(
        annual_total_potential_gpp=data_fapar_limitation[assim_var],
        annual_mean_ca=data_fapar_limitation["annual_mean_ca_in_GS"],
        annual_mean_chi=data_fapar_limitation["annual_mean_chi_in_GS"],
        annual_mean_vpd=data_fapar_limitation["annual_mean_VPD_in_GS"],
        annual_total_precip=data_fapar_limitation["annual_precip_molar"],
        annual_growing_season_length=data_fapar_limitation["N_growing_days"],
        years=data_fapar_limitation["year"].astype(str).astype("datetime64[Y]"),
        method=method,
        aridity_index=site_data["AI_from_cruts"],  # Not used by zhu method.
    )

    assert_allclose(
        data_fapar_limitation[f"fapar_max_{timescale_abbr}"],
        faparlim.fapar_max,
        rtol=1e-6,
    )
    assert_allclose(
        data_fapar_limitation[f"lai_max_{timescale_abbr}"],
        faparlim.lai_max,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    argnames="timescale,timescale_abbr,assim_var",
    argvalues=(
        pytest.param("fortnightly", "ft", "annual_total_A0", id="fortnightly"),
        pytest.param("subdaily", "hh", "annual_total_A0_smstress", id="subdaily"),
    ),
)
@pytest.mark.parametrize(
    argnames="method_predictions_dir, fapar_method, pheno_method",
    argvalues=(
        pytest.param("cai_zhou_method", "cai", "zhou", id="cai_zhou"),
        # pytest.param("zhu_method", "zhu", id="zhu"),
    ),
)
def test_phenology(
    site_data,
    data_fapar_limitation,
    data_phenology,
    timescale,  # parameterises data_* fixtures
    timescale_abbr,
    assim_var,
    method_predictions_dir,  # parameterises data_* fixtures
    fapar_method,
    pheno_method,
):
    """Regression test for FaparLimitation constructor with fortnightly data."""

    from pyrealm.phenology.fapar_limitation_new import FaparLimitationNew
    from pyrealm.phenology.phenology_new import Phenology

    faparlim = FaparLimitationNew(
        annual_total_potential_gpp=data_fapar_limitation[assim_var],
        annual_mean_ca=data_fapar_limitation["annual_mean_ca_in_GS"],
        annual_mean_chi=data_fapar_limitation["annual_mean_chi_in_GS"],
        annual_mean_vpd=data_fapar_limitation["annual_mean_VPD_in_GS"],
        annual_total_precip=data_fapar_limitation["annual_precip_molar"],
        annual_growing_season_length=data_fapar_limitation["N_growing_days"],
        years=data_fapar_limitation["year"].astype(str).astype("datetime64[Y]"),
        method=fapar_method,
        aridity_index=site_data["AI_from_cruts"],  # Not used by zhu method.
    )

    pheno = Phenology(
        daily_gpp=data_phenology["daily_A0"],
        datetimes=data_phenology["time"].astype("datetime64[D]"),
        fapar_limitation=faparlim,
    )

    # Fortnightly data is truncated by the last fortnight so need to truncate to match

    # Check the LAI time series to tolerance of data in file.
    assert_allclose(
        pheno.steady_state_lai,
        data_phenology[f"Ls_daily_{timescale_abbr}"][: len(pheno.steady_state_lai)],
        atol=1e-8,
    )
    assert_allclose(
        pheno.realised_lai,
        data_phenology[f"Ls_daily_lagged_{timescale_abbr}"][
            : len(pheno.steady_state_lai)
        ],
        atol=1e-8,
    )
