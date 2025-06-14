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
def splash_data():
    """Load the input data from nc file into xarray."""

    import xarray as xr

    datafile = (
        resources.files("pyrealm_build_data.phenology")
        / "DE_gri_splash_cru_ts4.07_2000_2019.nc"
    )

    splash_data = xr.load_dataset(datafile)

    return splash_data


@pytest.fixture()
def annual_fortnightly_data():
    """Load the input data from csv file."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.fortnightly_example")
        / "annual_outputs.csv"
    )

    return pd.read_csv(datafile)


@pytest.fixture()
def annual_subdaily_data():
    """Load the input data from csv file."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.subdaily_example")
        / "annual_outputs.csv"
    )

    return pd.read_csv(datafile)


def test_faparlimitation(site_data, annual_fortnightly_data):
    """Regression test for FaparLimitation constructor."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation

    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_fortnightly_data["ann_total_A0"].to_numpy(),
        annual_mean_ca=annual_fortnightly_data["annual_mean_ca_in_GS"].to_numpy(),
        annual_mean_chi=annual_fortnightly_data["annual_mean_chi_in_GS"].to_numpy(),
        annual_mean_vpd=annual_fortnightly_data["annual_mean_VPD_in_GS"].to_numpy(),
        annual_total_precip=annual_fortnightly_data["annual_precip_molar"].to_numpy(),
        aridity_index=site_data["AI_from_cruts"],
    )

    assert_allclose(
        annual_fortnightly_data["fapar_max"].to_numpy(), faparlim.fapar_max, rtol=1e-6
    )
    assert_allclose(
        annual_fortnightly_data["lai_max"].to_numpy(), faparlim.lai_max, rtol=1e-6
    )


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


@pytest.fixture()
def subdaily_data():
    """Load the input data from data file."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.subdaily_example")
        / "half_hourly_data.csv"
    )

    # Load the half hourly data
    subdaily_data = pd.read_csv(datafile)

    return subdaily_data


@pytest.fixture()
def daily_data():
    """Load the input data from data file."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.subdaily_example")
        / "daily_outputs.csv"
    )

    # Load the daily data
    daily_data = pd.read_csv(datafile)

    return daily_data


def test_faparlimitation_frompmodel_fortnightly(
    annual_fortnightly_data, site_data, fortnightly_data
):
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

    # Check the GPP predictions
    assert_allclose(pmodel.gpp, fortnightly_data["gpp"], rtol=1e-6)

    faparlim = FaparLimitation.from_pmodel(
        pmodel=pmodel,
        growing_season=fortnightly_data["growing_season"].to_numpy(),
        datetimes=fortnightly_data["time"].to_numpy(),
        precip=fortnightly_data["precip_molar_sum"].to_numpy(),
        aridity_index=site_data["AI_from_cruts"],
        gpp_penalty_factor=None,
    )

    assert_allclose(
        annual_fortnightly_data["fapar_max"].to_numpy(), faparlim.fapar_max, rtol=1e-6
    )
    assert_allclose(
        annual_fortnightly_data["lai_max"].to_numpy(), faparlim.lai_max, rtol=1e-6
    )


def test_faparlimitation_frompmodel_subdaily(
    site_data, subdaily_data, annual_subdaily_data
):
    """Regression test for from_subdailypmodel FaparLimitation class method."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation
    from pyrealm.pmodel import AcclimationModel, PModelEnvironment, SubdailyPModel

    env = PModelEnvironment(
        tc=subdaily_data["tc"].to_numpy(),
        vpd=subdaily_data["vpd"].to_numpy(),
        co2=subdaily_data["co2"].to_numpy(),
        patm=subdaily_data["patm"].to_numpy(),
        fapar=np.ones_like(subdaily_data["tc"]),
        ppfd=subdaily_data["ppfd"].to_numpy(),
    )

    datetimes = subdaily_data["time"].to_numpy().astype("datetime64[ns]")

    # Set up the datetimes of the observations and set the acclimation window
    acclim = AcclimationModel(
        datetimes=datetimes,
        alpha=1 / 15,
    )
    acclim.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Fit the subdaily potential GPP: fAPAR = 1 as set above and phi0 = 1/8
    subdaily_pmodel = SubdailyPModel(
        env=env,
        acclim_model=acclim,
        reference_kphio=1 / 8,
        method_kphio="temperature",
    )

    # Check the GPP predictions
    assert_allclose(subdaily_pmodel.gpp, subdaily_data["PMod_gpp"], rtol=1e-6)

    faparlim = FaparLimitation.from_pmodel(
        pmodel=subdaily_pmodel,
        growing_season=subdaily_data["growing_day"],
        datetimes=datetimes,
        precip=subdaily_data["precip_molar"],
        aridity_index=site_data["AI_from_cruts"],
        gpp_penalty_factor=subdaily_data["soilm_stress"],
    )

    assert_allclose(annual_subdaily_data["lai_max"], faparlim.lai_max, rtol=1e-6)
    assert_allclose(annual_subdaily_data["fapar_max"], faparlim.fapar_max, rtol=1e-6)
