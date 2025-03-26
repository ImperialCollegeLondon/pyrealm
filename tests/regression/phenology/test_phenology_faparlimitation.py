"""Test the FaparLimitation class."""

import json
from importlib import resources

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def site_data():
    """Load the site data."""

    datafile = resources.files("pyrealm_build_data") / "phenology/DE-GRI_site_data.json"

    with open(datafile) as json_src:
        site_data = json.load(json_src)

    return site_data


@pytest.fixture()
def annual_data():
    """Load the input data from netcdf file."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.fortnightly_example")
        / "annual_outputs.csv"
    )

    return pd.read_csv(datafile)


def test_faparlimitation(site_data, annual_data):
    """Regression test for FaparLimitation constructor."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation

    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_data["ann_total_A0"].to_numpy(),
        annual_mean_ca=annual_data["ca"].to_numpy(),
        annual_mean_chi=annual_data["chi"].to_numpy(),
        annual_mean_vpd=annual_data["vpd_mean"].to_numpy(),
        annual_total_precip=annual_data["precip_molar_sum"].to_numpy(),
        aridity_index=site_data["AI_from_cruts"],
    )

    assert_allclose(annual_data["fapar_max"].to_numpy(), faparlim.fapar_max, rtol=1e-6)
    assert_allclose(annual_data["lai_max"].to_numpy(), faparlim.lai_max, rtol=1e-6)


@pytest.fixture()
def fortnightly_data():
    """Load the input data for the from_pmodel class function from netcdf file."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.fortnightly_example")
        / "fortnightly_data.csv"
    )

    data = pd.read_csv(datafile)

    data["time"] = pd.to_datetime(data["time"])

    return data


@pytest.mark.skip("Need to expand the time handling to cope with datetimes >= 1 day")
def test_faparlimitation_frompmodel(annual_data, site_data, fortnightly_data):
    """Regression test for from_pmodel FaparLimitation class method."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation
    from pyrealm.pmodel import PModel, PModelEnvironment

    env = PModelEnvironment(
        tc=fortnightly_data["tc_mean"].to_numpy(),
        vpd=fortnightly_data["vpd_mean"].to_numpy(),
        co2=fortnightly_data["co2_mean"].to_numpy(),
        patm=fortnightly_data["patm_mean"].to_numpy(),
        fapar=np.ones_like(fortnightly_data["tc_mean"]),
        ppfd=fortnightly_data["ppfd_mean"].to_numpy(),
    )

    pmodel = PModel(
        env=env,
        reference_kphio=1 / 8,
        method_kphio="temperature",
    )

    faparlim = FaparLimitation.from_pmodel(
        pmodel=pmodel,
        growing_season=fortnightly_data["growing_season"].to_numpy(),
        datetimes=fortnightly_data["time"].to_numpy(),
        precip=fortnightly_data["precip_molar_sum"].to_numpy(),
        aridity_index=site_data["AI_from_cruts"],
    )

    assert np.allclose(annual_data["fapar_max"].to_numpy(), faparlim.fapar_max)
    assert np.allclose(annual_data["lai_max"].to_numpy(), faparlim.lai_max)
