"""Shared fixtures for phenology testing."""

import json
from importlib import resources

import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def de_gri_half_hourly_inputs():
    """Loads the half hourly fluxnet data."""

    # Load the data
    datapath = (
        resources.files("pyrealm_build_data.phenology") / "DE_GRI_hh_fluxnet_simple.csv"
    )

    de_gri_data = pd.read_csv(datapath, na_values=[-9999, -9999.0])

    # Calculate time as np.datetime64 and scale other vars
    de_gri_data["time"] = pd.to_datetime(
        de_gri_data["TIMESTAMP_START"], format="%Y%m%d%H%M"
    )

    # # Blank out temperatures under 25°C
    # de_gri_data["TA_F"] = de_gri_data["VPD_F"].where(de_gri_data["VPD_F"] >= -25)

    # # VPD from hPa to Pa
    de_gri_data["VPD_F"] = de_gri_data["VPD_F"].to_numpy() * 100
    # Pressure from kPa to Pa
    de_gri_data["PA_F"] = de_gri_data["PA_F"].to_numpy() * 1000
    # PPFD from SWDOWN
    de_gri_data["PPFD"] = de_gri_data["SW_IN_F_MDS"].to_numpy() * 2.04

    return de_gri_data


@pytest.fixture
def de_gri_splash_inputs():
    """Loads the daily splash inputs."""

    # Load the data
    datapath = (
        resources.files("pyrealm_build_data.phenology")
        / "DE_gri_splash_cru_ts4.07_2000_2019.nc"
    )

    de_gri_data = xr.load_dataset(datapath)

    return de_gri_data


@pytest.fixture
def de_gri_hh_outputs():
    """Loads the half hourly phenology outputs."""

    # Load the data
    datapath = resources.files("pyrealm_build_data.phenology") / "python_hh_outputs.csv"

    de_gri_data = pd.read_csv(datapath, na_values=[-9999])

    # Calculate time as np.datetime64 and scale other vars
    de_gri_data["time"] = pd.to_datetime(
        de_gri_data["time"], format="%Y-%m-%d %H:%M:%S"
    )

    return de_gri_data


@pytest.fixture
def de_gri_daily_outputs():
    """Loads the daily phenology data."""

    # Load the data
    datapath = (
        resources.files("pyrealm_build_data.phenology") / "python_daily_outputs.csv"
    )

    de_gri_data = pd.read_csv(datapath, na_values=[-9999])

    # Calculate time as np.datetime64 and scale other vars
    de_gri_data["date"] = pd.to_datetime(de_gri_data["time"], format="%Y-%m-%d")

    return de_gri_data


@pytest.fixture
def de_gri_constants():
    """Load the site constants."""

    # Load the data
    datapath = resources.files("pyrealm_build_data.phenology") / "DE-GRI_site_data.json"

    with open(datapath, "rb") as site_file:
        constants = json.load(site_file)

    return constants
