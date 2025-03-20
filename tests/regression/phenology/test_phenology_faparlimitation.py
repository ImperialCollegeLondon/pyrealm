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
        resources.files("pyrealm_build_data") / "phenology/python_annual_outputs.csv"
    )

    return pd.read_csv(datafile)


def test_faparlimitation(site_data, annual_data):
    """Regression test for FaparLimitation constructor."""

    from pyrealm.constants import CoreConst
    from pyrealm.phenology.fapar_limitation import FaparLimitation

    core_const = CoreConst()

    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_data[
            "annual_total_A0_subdaily_with_water_stress"
        ].to_numpy()
        / core_const.k_c_molmass,
        annual_mean_ca=annual_data["annual_mean_ca_in_GS"].to_numpy(),
        annual_mean_chi=annual_data["annual_mean_chi_in_GS"].to_numpy(),
        annual_mean_vpd=annual_data["annual_mean_VPD_in_GS"].to_numpy(),
        annual_total_precip=annual_data["annual_precip_molar"].to_numpy(),
        aridity_index=site_data["AI_from_cruts"],
    )

    assert_allclose(annual_data["fapar_max"].to_numpy(), faparlim.fapar_max, rtol=1e-6)
    assert_allclose(annual_data["lai_max"].to_numpy(), faparlim.lai_max, rtol=1e-6)


@pytest.fixture()
def pmodel_data():
    """Load the input data for the from_pmodel class function from netcdf file."""

    datafile_daily = (
        resources.files("pyrealm_build_data") / "phenology/python_daily_outputs.csv"
    )

    daily_data = pd.read_csv(datafile_daily)

    datafile_halfhourly = (
        resources.files("pyrealm_build_data") / "phenology/python_hh_outputs.csv"
    )

    hh_data = pd.read_csv(datafile_halfhourly)

    return (daily_data, hh_data)

    tc = np.array(hh_data["ta"])
    vpd = np.array(hh_data["vpd"])
    co2 = np.array(hh_data["co2"])
    patm = np.array(hh_data["pa_f"])
    growing_season = np.array(daily_data["growing_day_filtered"])
    datetimes = np.array(hh_data["time"], dtype="datetime64")
    precipitation = np.array(daily_data["pre"])
    ppfd = np.array(hh_data["ppfd"])
    fapar_max = np.array(np.unique(daily_data["annual_fapar_max"]))
    lai_max = np.array(np.unique(daily_data["annual_lai_max"]))

    return (
        tc,
        vpd,
        co2,
        patm,
        growing_season,
        datetimes,
        precipitation,
        ppfd,
        fapar_max,
        lai_max,
    )


def test_faparlimitation_frompmodel(annual_data, site_data, pmodel_data):
    """Regression test for from_pmodel FaparLimitation class method."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation
    from pyrealm.pmodel import PModel, PModelEnvironment

    daily_data, hh_data = pmodel_data

    env = PModelEnvironment(
        tc=hh_data["ta"].to_numpy(),
        vpd=hh_data["vpd"].to_numpy(),
        co2=hh_data["co2"].to_numpy(),
        patm=hh_data["pa_f"].to_numpy(),
        fapar=np.ones_like(hh_data["ta"]),
        ppfd=hh_data["ppfd"].to_numpy(),
    )

    pmodel = PModel(
        env=env,
        reference_kphio=1 / 8,
        method_kphio="temperature",
    )

    faparlim = FaparLimitation.from_pmodel(
        pmodel=pmodel,
        growing_season=daily_data["growing_day_filtered"],
        datetimes=hh_data["time"],
        precip=daily_data["pre"],
        aridity_index=site_data["AI_from_cruts"],
    )

    assert np.allclose(annual_data["fapar_max"].to_numpy(), faparlim.fapar_max)
    assert np.allclose(annual_data["lai_max"].to_numpy(), faparlim.lai_max)
