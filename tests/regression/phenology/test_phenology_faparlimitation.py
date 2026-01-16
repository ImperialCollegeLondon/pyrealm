"""Test the FaparLimitation class."""

import json
from importlib import resources

import numpy as np
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

    datafile = (
        resources.files("pyrealm_build_data.phenology.fortnightly_example")
        / "annual_outputs.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def annual_subdaily_data():
    """Load the input data from csv file."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.subdaily_example")
        / "annual_outputs.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def fortnightly_data():
    """Load the input data for the from_pmodel class function from netcdf file."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.fortnightly_example")
        / "fortnightly_data.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def subdaily_data():
    """Load the input data from data file."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.subdaily_example")
        / "half_hourly_data.csv"
    )

    # Load the half hourly data
    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def subdaily_phenology():
    """Daily expected phenology from the fortnightly example."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.subdaily_example")
        / "daily_outputs.csv"
    )

    # Load the daily phenology data
    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture
def fortnightly_phenology():
    """Daily expected phenology from the fortnightly example."""
    datafile = (
        resources.files("pyrealm_build_data.phenology.fortnightly_example")
        / "daily_outputs.csv"
    )

    data = dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))

    data["time"] = data["time"].astype("datetime64[D]")
    return data


def test_faparlimitation_fortnightly(site_data, annual_fortnightly_data):
    """Regression test for FaparLimitation constructor with fortnightly data."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation

    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_fortnightly_data["ann_total_A0"],
        annual_mean_ca=annual_fortnightly_data["annual_mean_ca_in_GS"],
        annual_mean_chi=annual_fortnightly_data["annual_mean_chi_in_GS"],
        annual_mean_vpd=annual_fortnightly_data["annual_mean_VPD_in_GS"],
        annual_total_precip=annual_fortnightly_data["annual_precip_molar"],
        annual_growing_season_length=annual_fortnightly_data["N_growing_days"],
        years=annual_fortnightly_data["time"].astype("datetime64[Y]"),
        aridity_index=site_data["AI_from_cruts"],
    )

    assert_allclose(annual_fortnightly_data["fapar_max"], faparlim.fapar_max, rtol=1e-6)
    assert_allclose(annual_fortnightly_data["lai_max"], faparlim.lai_max, rtol=1e-6)

    assert_allclose(
        annual_fortnightly_data["m"], faparlim.lai_to_gpp_ratio_m, rtol=1e-6
    )


def test_faparlimitation_subdaily(site_data, annual_subdaily_data):
    """Regression test for FaparLimitation constructor using subdaily inputs."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation

    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_subdaily_data[
            "ann_total_A0_subdaily_smstress"
        ],
        annual_mean_ca=annual_subdaily_data["annual_mean_ca_in_GS"],
        annual_mean_chi=annual_subdaily_data["annual_mean_chi_in_GS"],
        annual_mean_vpd=annual_subdaily_data["annual_mean_VPD_in_GS"],
        annual_total_precip=annual_subdaily_data["annual_precip_molar"],
        annual_growing_season_length=annual_subdaily_data["N_growing_days"],
        years=annual_subdaily_data["time"].astype("datetime64[Y]"),
        aridity_index=site_data["AI_from_cruts"],
    )

    assert_allclose(annual_subdaily_data["fapar_max"], faparlim.fapar_max, rtol=1e-6)
    assert_allclose(annual_subdaily_data["lai_max"], faparlim.lai_max, rtol=1e-6)

    assert_allclose(annual_subdaily_data["m"], faparlim.lai_to_gpp_ratio_m, rtol=1e-6)


def test_phenology_subdaily(site_data, annual_subdaily_data, subdaily_phenology):
    """Regression test of the Phenology class on subdaily data."""
    from pyrealm.phenology.fapar_limitation import FaparLimitation, Phenology

    annual_subdaily_data["time"] = (
        annual_subdaily_data["time"].astype(str).astype("datetime64[Y]")
    )

    # Create fapar limitation - this is tested separately
    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_subdaily_data[
            "ann_total_A0_subdaily_smstress"
        ],
        annual_mean_ca=annual_subdaily_data["annual_mean_ca_in_GS"],
        annual_mean_chi=annual_subdaily_data["annual_mean_chi_in_GS"],
        annual_mean_vpd=annual_subdaily_data["annual_mean_VPD_in_GS"],
        annual_total_precip=annual_subdaily_data["annual_precip_molar"],
        annual_growing_season_length=annual_subdaily_data["N_growing_days"],
        years=annual_subdaily_data["time"],
        aridity_index=site_data["AI_from_cruts"],
    )

    pheno = Phenology(
        daily_gpp=subdaily_phenology["daily_A0"],
        datetimes=subdaily_phenology["time"],
        fapar_limitation=faparlim,
    )

    # Check the LAI time series to tolerance of data in file.
    assert_allclose(pheno.steady_state_LAI, subdaily_phenology["Ls_daily"], atol=1e-5)
    assert_allclose(
        pheno.realised_LAI, subdaily_phenology["Ls_daily_lagged"], atol=1e-5
    )


def test_phenology_fortnightly(
    site_data, annual_fortnightly_data, fortnightly_phenology
):
    """Regression test of the Phenology class on subdaily data."""
    from pyrealm.phenology.fapar_limitation import FaparLimitation, Phenology

    annual_fortnightly_data["time"] = (
        annual_fortnightly_data["time"].astype(str).astype("datetime64[Y]")
    )

    # Create fapar limitation - this is tested separately
    faparlim = FaparLimitation(
        annual_total_potential_gpp=annual_fortnightly_data["ann_total_A0"],
        annual_mean_ca=annual_fortnightly_data["annual_mean_ca_in_GS"],
        annual_mean_chi=annual_fortnightly_data["annual_mean_chi_in_GS"],
        annual_mean_vpd=annual_fortnightly_data["annual_mean_VPD_in_GS"],
        annual_total_precip=annual_fortnightly_data["annual_precip_molar"],
        annual_growing_season_length=annual_fortnightly_data["N_growing_days"],
        years=annual_fortnightly_data["time"],
        aridity_index=site_data["AI_from_cruts"],
    )

    pheno = Phenology(
        daily_gpp=fortnightly_phenology["daily_A0"],
        datetimes=fortnightly_phenology["time"],
        fapar_limitation=faparlim,
    )

    # Check the LAI time series to tolerance of data in file.
    assert_allclose(
        pheno.steady_state_LAI, fortnightly_phenology["Ls_daily"], atol=1e-5
    )
    assert_allclose(
        pheno.realised_LAI, fortnightly_phenology["Ls_daily_lagged"], atol=1e-5
    )


def test_faparlimitation_and_phenology_frompmodel_fortnightly(
    annual_fortnightly_data, site_data, fortnightly_data, fortnightly_phenology
):
    """Regression test for from_pmodel FaparLimitation class method and Phenology."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation, Phenology
    from pyrealm.pmodel import PModel, PModelEnvironment

    fortnightly_data["time"] = fortnightly_data["time"].astype("datetime64[D]")

    env = PModelEnvironment(
        tc=fortnightly_data["tc_mean"],
        vpd=fortnightly_data["vpd_mean"],
        co2=fortnightly_data["co2_mean"],
        patm=fortnightly_data["patm_mean"],
        fapar=np.ones_like(fortnightly_data["tc_mean"]),
        ppfd=fortnightly_data["ppfd_mean"],
    )

    fortnightly_pmodel = PModel(
        env=env,
        reference_kphio=1 / 8,
        method_kphio="temperature",
    )

    # Check the GPP predictions
    assert_allclose(fortnightly_pmodel.gpp, fortnightly_data["gpp"], rtol=1e-6)

    faparlim = FaparLimitation.from_pmodel(
        pmodel=fortnightly_pmodel,
        growing_season=fortnightly_data["growing_season"],
        datetimes=fortnightly_data["time"],
        precip=fortnightly_data["precip_molar_sum"],
        aridity_index=site_data["AI_from_cruts"],
        gpp_penalty_factor=None,
    )

    assert_allclose(annual_fortnightly_data["fapar_max"], faparlim.fapar_max, rtol=1e-6)
    assert_allclose(annual_fortnightly_data["lai_max"], faparlim.lai_max, rtol=1e-6)
    assert_allclose(
        annual_fortnightly_data["m"], faparlim.lai_to_gpp_ratio_m, rtol=1e-6
    )

    days, gpp = fortnightly_pmodel._get_daily_gpp(datetimes=fortnightly_data["time"])

    # Scale daily GPP in µmol m2 s up to daily molar assimilation.
    daily_A0 = (gpp * [60 * 60 * 24 * 1e-6]) / fortnightly_pmodel.core_const.k_c_molmass

    assert_allclose(daily_A0, fortnightly_phenology["daily_A0"], atol=1e-5)

    pheno = Phenology(daily_gpp=daily_A0, datetimes=days, fapar_limitation=faparlim)

    assert_allclose(
        pheno.steady_state_LAI, fortnightly_phenology["Ls_daily"], atol=1e-5
    )
    assert_allclose(
        pheno.realised_LAI, fortnightly_phenology["Ls_daily_lagged"], atol=1e-5
    )


def test_faparlimitation_and_phenology_frompmodel_subdaily(
    site_data,
    subdaily_data,
    annual_subdaily_data,
    subdaily_phenology,
):
    """Regression test for from_subdailypmodel FaparLimitation class method."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation, Phenology
    from pyrealm.pmodel import AcclimationModel, PModelEnvironment, SubdailyPModel

    subdaily_data["time"] = subdaily_data["time"].astype("datetime64[ns]")

    # PATM is read in as integer - and this isn't compatible with the SubdailyPModel,
    # because it does not support np.nan values.
    env = PModelEnvironment(
        tc=subdaily_data["tc"],
        vpd=subdaily_data["vpd"],
        co2=subdaily_data["co2"],
        patm=subdaily_data["patm"].astype(float),
        fapar=np.ones_like(subdaily_data["tc"]),
        ppfd=subdaily_data["ppfd"],
    )

    # Set up the datetimes of the observations and set the acclimation window
    acclim = AcclimationModel(
        datetimes=subdaily_data["time"],
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

    # Does not require datetimes - taken from acclimation model
    faparlim = FaparLimitation.from_pmodel(
        pmodel=subdaily_pmodel,
        growing_season=subdaily_data["growing_day"],
        precip=subdaily_data["precip_molar"],
        aridity_index=site_data["AI_from_cruts"],
        gpp_penalty_factor=subdaily_data["soilm_stress"],
    )

    assert_allclose(annual_subdaily_data["lai_max"], faparlim.lai_max, rtol=1e-6)
    assert_allclose(annual_subdaily_data["fapar_max"], faparlim.fapar_max, rtol=1e-6)
    assert_allclose(annual_subdaily_data["m"], faparlim.lai_to_gpp_ratio_m, rtol=1e-6)

    days, gpp = subdaily_pmodel._get_daily_gpp()

    # Scale daily GPP in µmol m2 s up to daily molar assimilation.
    daily_A0 = (gpp * [60 * 60 * 24 * 1e-6]) / subdaily_pmodel.core_const.k_c_molmass

    assert_allclose(daily_A0, subdaily_phenology["daily_A0"], atol=1e-5)

    pheno = Phenology(daily_gpp=daily_A0, datetimes=days, fapar_limitation=faparlim)

    assert_allclose(pheno.steady_state_LAI, subdaily_phenology["Ls_daily"], atol=1e-5)
    assert_allclose(
        pheno.realised_LAI, subdaily_phenology["Ls_daily_lagged"], atol=1e-5
    )
