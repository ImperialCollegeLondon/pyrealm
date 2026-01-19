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

    datafile = resources.files("pyrealm_build_data") / "phenology/DE-GRI_site_data.json"

    with open(datafile) as json_src:
        site_data = json.load(json_src)

    return site_data


@pytest.fixture()
def annual_fortnightly_data():
    """Load the input data from csv file."""

    datafile = pd.read_csv(
        str(
            resources.files("pyrealm_build_data.phenology.fortnightly_example")
            / "annual_outputs.csv"
        )
    )

    zhu = pd.read_csv(
        str(
            resources.files("pyrealm_build_data.phenology.zhu_method")
            / "zhu_annual_fapar_max_from_fortnightly_data.csv"
        )
    )

    data = datafile.merge(zhu, left_on="time", right_on="year")
    return dataframe_to_dict_of_nparrays(data)


@pytest.mark.parametrize(
    argnames="method",
    argvalues=(pytest.param("cai", id="cai"), pytest.param("zhu", id="zhu")),
)
def test_faparlimitation_fortnightly(site_data, annual_fortnightly_data, method):
    """Regression test for FaparLimitation constructor with fortnightly data."""

    from pyrealm.phenology.fapar_limitation_new import FaparLimitation

    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_fortnightly_data["ann_total_A0"],
        annual_mean_ca=annual_fortnightly_data["annual_mean_ca_in_GS"],
        annual_mean_chi=annual_fortnightly_data["annual_mean_chi_in_GS"],
        annual_mean_vpd=annual_fortnightly_data["annual_mean_VPD_in_GS"],
        annual_total_precip=annual_fortnightly_data["annual_precip_molar"],
        annual_growing_season_length=annual_fortnightly_data["N_growing_days"],
        years=annual_fortnightly_data["time"].astype("datetime64[Y]"),
        method=method,
        aridity_index=site_data["AI_from_cruts"],
    )

    # temp fix on naming
    target = "fapar_max" if method == "cai" else "zhu_fapar_max"

    assert_allclose(annual_fortnightly_data[target], faparlim.fapar_max, rtol=1e-6)
    # assert_allclose(annual_fortnightly_data["lai_max"], faparlim.lai_max, rtol=1e-6)

    # assert_allclose(
    #     annual_fortnightly_data["m"], faparlim.lai_to_gpp_ratio_m, rtol=1e-6
    # )
