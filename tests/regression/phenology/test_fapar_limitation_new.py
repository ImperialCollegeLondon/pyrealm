"""Test the FaparLimitation class."""

import json
from importlib import resources

import pandas as pd
import pytest
from numpy.testing import assert_allclose


def dataframe_to_dict_of_nparrays(data):
    """Utility function to preconvert a dataframe of pd.Series to np.array."""

    data_dict = {k: v.to_numpy() for k, v in data.items()}

    return data_dict


@pytest.fixture
def site_data():
    """Load the site data."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.inputs.source")
        / "DE-GRI_site_data.json"
    )

    with open(str(datafile)) as json_src:
        site_data = json.load(json_src)

    return site_data


@pytest.fixture()
def data_fapar_limitation(timescale, method_predictions_dir):
    """Load the input data from csv file."""

    inputs = pd.read_csv(
        str(
            resources.files(f"pyrealm_build_data.phenology.inputs.{timescale}")
            / "annual_inputs.csv"
        )
    )

    predictions = pd.read_csv(
        str(
            resources.files(f"pyrealm_build_data.phenology.{method_predictions_dir}")
            / "fapar_max_predictions.csv"
        )
    )

    data = inputs.merge(predictions)
    return dataframe_to_dict_of_nparrays(data)


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

    from pyrealm.phenology.fapar_limitation_new import FaparLimitation

    faparlim = FaparLimitation(
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
