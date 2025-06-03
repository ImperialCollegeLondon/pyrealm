"""Shared fixtures for phenology testing."""

import json
from importlib import resources

import pandas as pd
import pytest
import xarray


@pytest.fixture
def de_gri_splash_data():
    """Loads the splash data."""

    # Load the data
    datapath = (
        resources.files("pyrealm_build_data.phenology")
        / "DE_gri_splash_cru_ts4.07_2000_2019.nc"
    )

    return xarray.load_dataset(datapath)


@pytest.fixture
def de_gri_subdaily_data():
    """Loads the half hourly data."""

    # Load the data
    datapath = (
        resources.files("pyrealm_build_data.phenology.subdaily_example")
        / "half_hourly_data.csv"
    )

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
        resources.files("pyrealm_build_data.phenology.subdaily_example")
        / "daily_outputs.csv"
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
