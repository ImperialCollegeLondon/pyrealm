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
def annual_data():
    """Load the input data from csv file."""

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


@pytest.fixture()
def subdaily_data():
    """Load the input data from data file."""

    from pyrealm.core.water import convert_water_mm_to_moles

    datafile = (
        resources.files("pyrealm_build_data.phenology") / "DE_GRI_hh_fluxnet_simple.csv"
    )

    # Load the half hourly data - ignoring mypy's dislike of perfectly functional
    # numeric inputs to na_values.
    de_gri_hh_pd = pd.read_csv(
        datafile,
        na_values=["-9999-9999.0", -9999.0, -9999],  # type: ignore[list-item]
    )

    # Calculate time as np.datetime64, set as the index and convert to xarray
    de_gri_hh_pd["time"] = pd.to_datetime(
        de_gri_hh_pd["TIMESTAMP_START"], format="%Y%m%d%H%M"
    )
    de_gri_hh_pd = de_gri_hh_pd.set_index("time")
    de_gri_hh_xr = de_gri_hh_pd.to_xarray()

    # # Blank out temperatures under 25°C
    de_gri_hh_xr["TA_F"] = de_gri_hh_xr["TA_F"].where(de_gri_hh_xr["TA_F"] >= -25)

    # # VPD from hPa to Pa
    de_gri_hh_xr["VPD_F"] = de_gri_hh_xr["VPD_F"] * 100
    # Pressure from kPa to Pa
    de_gri_hh_xr["PA_F"] = de_gri_hh_xr["PA_F"] * 1000
    # PPFD from SWDOWN
    de_gri_hh_xr["PPFD"] = de_gri_hh_xr["SW_IN_F_MDS"] * 2.04

    # Convert precipitation to molar values at half hour scale to aggregate up to annual
    # totals. Can't simply convert annual means here - need to convert with conditions
    # at half hourly time step.
    #
    # - Both FluxNET and CRU (loaded below for aridity and soil moisture calculations)
    #   provide precipitation data. CRU is more consistent with the aridity index
    #   calculation and hence f_0, but the FluxNET data is more site appropriate so is
    #   used here. We also need Temp and PATM to convert water mm to water mols, and
    #   currently the soil moisture inputs don't include that from the daily CRU data.

    # Calculate water as mols m2 not mm m2
    site_precip_molar = convert_water_mm_to_moles(
        water_mm=de_gri_hh_xr["P_F"].to_numpy(),
        tc=de_gri_hh_xr["TA_F"].to_numpy(),
        patm=de_gri_hh_xr["PA_F"].to_numpy(),
    )

    de_gri_hh_xr = de_gri_hh_xr.assign(P_F_MOLAR=("time", site_precip_molar))

    return de_gri_hh_xr


def test_faparlimitation_fromsubdailypmodel(site_data, splash_data, subdaily_data):
    """Regression test for from_subdailypmodel FaparLimitation class method."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation, daily_to_subdaily
    from pyrealm.pmodel import AcclimationModel, PModelEnvironment, SubdailyPModel
    from pyrealm.pmodel.functions import calc_soilmstress_mengoli

    env = PModelEnvironment(
        tc=subdaily_data["TA_F"].to_numpy(),
        vpd=subdaily_data["VPD_F"].to_numpy(),
        co2=subdaily_data["CO2_F_MDS"].to_numpy(),
        patm=subdaily_data["PA_F"].to_numpy(),
        fapar=np.ones_like(subdaily_data["TA_F"]),
        ppfd=subdaily_data["PPFD"].to_numpy(),
    )

    # Set up the datetimes of the observations and set the acclimation window
    acclim = AcclimationModel(
        datetimes=subdaily_data["time"].to_numpy(),
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

    # Reduce splash data down to the same datetimes as the other data
    splash_data = splash_data.sel(
        time=slice(subdaily_data["time"][0], subdaily_data["time"][-1])
    )

    aridity_index = splash_data["pet"].mean() / splash_data["pre"].mean()

    # Find growing season
    growing_season = subdaily_data["TA_F"] >= 0

    # Calculate soil moisture stress for gpp penalty
    daily_soil_moisture_stress = calc_soilmstress_mengoli(
        soilm=splash_data["wn"].to_numpy() / 150,
        aridity_index=float(aridity_index),
    )

    soil_moisture_stress = daily_to_subdaily(
        daily_soil_moisture_stress, subdaily_data["time"].data
    )

    faparlim = FaparLimitation.from_subdailypmodel(
        subdaily_pmodel,
        growing_season.data,
        subdaily_data["time"].data,
        subdaily_data["P_F_MOLAR"].data,
        aridity_index,
        soil_moisture_stress,
    )

    assert np.allclose(subdaily_data["fapar_max"].to_numpy(), faparlim.fapar_max)
    assert np.allclose(subdaily_data["lai_max"].to_numpy(), faparlim.lai_max)
