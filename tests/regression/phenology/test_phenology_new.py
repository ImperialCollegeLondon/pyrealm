"""Test the PhenologyNew class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def fapar_limitation_instance(
    site_data,
    annual_inputs,
    timescale,  # Also used to parameterize annual_inputs
    fapar_method,
):
    """Provides FaparLimitation instances for testing Phenology."""
    from pyrealm.phenology.fapar_limitation_new import FaparLimitationNew

    assim_var = "annual_total_A0" if timescale == "ft" else "annual_total_A0_smstress"

    return FaparLimitationNew(
        annual_total_potential_gpp=annual_inputs[assim_var],
        annual_mean_ca=annual_inputs["annual_mean_ca_in_GS"],
        annual_mean_chi=annual_inputs["annual_mean_chi_in_GS"],
        annual_mean_vpd=annual_inputs["annual_mean_VPD_in_GS"],
        annual_total_precip=annual_inputs["annual_precip_molar"],
        annual_growing_season_length=annual_inputs["N_growing_days"],
        years=annual_inputs["year"].astype(str).astype("datetime64[Y]"),
        method=fapar_method,
        aridity_index=site_data["AI_from_cruts"],  # Not used by zhu method.
    )


@pytest.mark.parametrize(
    argnames="method_predictions_dir, fapar_method, pheno_method",
    argvalues=(
        pytest.param("cai_zhou_method", "cai", "zhou", id="cai_zhou"),
        # pytest.param("zhu_method", "zhu", "zhu", id="zhu"),
    ),
)
@pytest.mark.parametrize(
    argnames="timescale",
    argvalues=(
        pytest.param("hh", id="subdaily"),
        pytest.param("ft", id="fortnightly"),
    ),
)
def test_phenology_cai_zhou(
    daily_assimilation,
    fapar_limitation_instance,
    daily_lai_predictions,
    method_predictions_dir,
    fapar_method,  # Parametrizes fapar_limitation_instance fixture
    pheno_method,
    # Timescale argument also parametrises fapar_limitation_instance, daily_assimilation
    # and daily_lai_predictions fixtures
    timescale,
):
    """Regression test for FaparLimitation constructor.

    This test compares the outputs of FaparLimitation for a given method against the
    predicted outputs from the reference code for each implementation.
    """

    from pyrealm.phenology.phenology_new import PhenologyNew

    kwargs = {"aet_pet_ratio": np.array([6])} if pheno_method == "zhu" else {}

    pheno = PhenologyNew(
        daily_gpp=daily_assimilation["daily_A0"],
        datetimes=daily_assimilation["time"].astype("datetime64[D]"),
        fapar_limitation=fapar_limitation_instance,
        method=pheno_method,
        **kwargs,
    )

    # Fortnightly data is truncated by the last fortnight so need to truncate to match

    # Check the LAI time series to tolerance of data in file.
    assert_allclose(
        pheno.steady_state_lai,
        daily_lai_predictions[f"Ls_daily_{timescale}"][: len(pheno.steady_state_lai)],
        atol=1e-8,
    )
    assert_allclose(
        pheno.realised_lai,
        daily_lai_predictions[f"Ls_daily_lagged_{timescale}"][
            : len(pheno.realised_lai)
        ],
        atol=1e-8,
    )


@pytest.mark.parametrize(
    argnames="method_predictions_dir, fapar_method, pheno_method",
    argvalues=(pytest.param("zhu_method", "zhu", "zhu", id="zhu"),),
)
@pytest.mark.parametrize(
    argnames="timescale",
    argvalues=(
        pytest.param("hh", id="subdaily"),
        pytest.param("ft", id="fortnightly"),
    ),
)
def test_phenology_zhu(
    site_data,
    daily_assimilation,
    fapar_limitation_instance,
    daily_lai_predictions,
    method_predictions_dir,
    fapar_method,  # Parametrizes fapar_limitation_instance fixture
    pheno_method,
    # Timescale argument also parametrises fapar_limitation_instance, daily_assimilation
    # and daily_lai_predictions fixtures
    timescale,
):
    """Regression test for PhenologyMethodZhu.

    This is separated from the test above because the original code implementation only
    handles a single year of values at a time, so the pyrealm implementation has to be
    run in annual blocks to match the original test.
    """

    from pyrealm.phenology.phenology_new import PhenologyNew

    steady_state_lai = []
    realised_lai = []

    year_values = daily_assimilation["time"].astype("datetime64[Y]")
    years = np.unique(year_values)

    for year in years:
        # Get indices to subset the data and change the spinup length between normal and
        # leap years to match the simple tiling in the original implementation.
        year_indices = np.where(year_values == year)[0]

        pheno = PhenologyNew(
            daily_gpp=daily_assimilation["daily_A0"][year_indices],
            datetimes=daily_assimilation["time"][year_indices].astype("datetime64[D]"),
            fapar_limitation=fapar_limitation_instance,
            method=pheno_method,
            aet_pet_ratio=np.array([site_data["aet_pet_ratio"]]),
            spinup_length=len(year_indices),
        )

        steady_state_lai.append(pheno.steady_state_lai)
        realised_lai.append(pheno.realised_lai)

    # Combine the annual blocked predictions back into the single time series
    steady_state_lai = np.concatenate(steady_state_lai)
    realised_lai = np.concatenate(realised_lai)

    # Check the LAI time series to tolerance of data in file.
    # Fortnightly data is truncated by the last fortnight so need to truncate to match
    assert_allclose(
        steady_state_lai,
        daily_lai_predictions[f"Ls_daily_{timescale}"][: len(steady_state_lai)],
        atol=1e-8,
    )
    assert_allclose(
        realised_lai,
        daily_lai_predictions[f"Ls_daily_lagged_{timescale}"][: len(realised_lai)],
        atol=1e-8,
    )
