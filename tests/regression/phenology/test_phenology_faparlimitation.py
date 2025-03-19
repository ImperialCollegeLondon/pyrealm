"""Test the FaparLimitation class."""

from importlib import resources

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def annual_data():
    """Load the input data from netcdf file."""

    datafile = (
        resources.files("pyrealm_build_data") / "phenology/python_annual_outputs.csv"
    )

    annual_data = pd.read_csv(datafile).to_dict(orient="list")

    annual_total_A0_subdaily = np.array(
        annual_data["annual_total_A0_subdaily_with_water_stress"]
    )
    annual_total_P = np.array(annual_data["annual_precip_molar"])
    annual_mean_ca = np.array(annual_data["annual_mean_ca_in_GS"])
    annual_mean_chi = np.array(annual_data["annual_mean_chi_in_GS"])
    annual_mean_vpd = np.array(annual_data["annual_mean_VPD_in_GS"])
    fapar_max = np.array(annual_data["fapar_max"])
    lai_max = np.array(annual_data["lai_max"])

    return (
        annual_total_A0_subdaily,
        annual_total_P,
        annual_mean_ca,
        annual_mean_chi,
        annual_mean_vpd,
        fapar_max,
        lai_max,
    )


def test_faparlimitation(annual_data):
    """Regression test for FaparLimitation constructor."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation

    (
        annual_total_A0_subdaily,
        annual_total_P,
        annual_mean_ca,
        annual_mean_chi,
        annual_mean_vpd,
        exp_faparmax,
        exp_laimax,
    ) = annual_data

    aridity_index = np.array(1.17225709)

    faparlim = FaparLimitation(
        annual_total_A0_subdaily,
        annual_mean_ca,
        annual_mean_chi,
        annual_mean_vpd,
        annual_total_P,
        aridity_index,
    )

    assert np.allclose(exp_faparmax, faparlim.fapar_max)
    assert np.allclose(exp_laimax, faparlim.lai_max)


@pytest.fixture()
def pmodel_data():
    """Load the input data for the from_pmodel class function from netcdf file."""

    datafile_daily = (
        resources.files("pyrealm_build_data") / "phenology/python_daily_outputs.csv"
    )

    daily_data = pd.read_csv(datafile_daily).to_dict(orient="list")

    datafile_halfhourly = (
        resources.files("pyrealm_build_data") / "phenology/python_hh_outputs.csv"
    )

    hh_data = pd.read_csv(datafile_halfhourly).to_dict(orient="list")

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


def test_faparlimitation_frompmodel(pmodel_data):
    """Regression test for from_pmodel FaparLimitation class method."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation
    from pyrealm.pmodel import PModel, PModelEnvironment

    (
        tc,
        vpd,
        co2,
        patm,
        growing_season,
        datetimes,
        precipitation,
        ppfd,
        exp_faparmax,
        exp_laimax,
    ) = pmodel_data

    env = PModelEnvironment(
        tc=tc, vpd=vpd, co2=co2, patm=patm, fapar=np.ones_like(tc), ppfd=ppfd
    )

    pmodel = PModel(
        env=env,
        reference_kphio=1 / 8,
        method_kphio="temperature",
    )

    aridity_index = np.array(1.17225709)

    faparlim = FaparLimitation.from_pmodel(
        pmodel, growing_season, datetimes, precipitation, aridity_index
    )

    assert np.allclose(exp_faparmax, faparlim.fapar_max)
    assert np.allclose(exp_laimax, faparlim.lai_max)
