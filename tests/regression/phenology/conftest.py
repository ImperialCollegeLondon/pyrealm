"""Shared fixtures for phenology testing."""

from importlib import resources

import pandas as pd
import pytest


@pytest.fixture
def de_gri_half_hourly_data():
    """Loads the half hourly fluxnet data."""

    # Load the data
    datapath = (
        resources.files("pyrealm_build_data.phenology")
        / "FLX_DE-Gri_FLUXNET2015_phenology_inputs.csv"
    )

    de_gri_data = pd.read_csv(datapath, na_values=[-9999])

    # Calculate time as np.datetime64 and scale other vars
    de_gri_data["time"] = pd.to_datetime(
        de_gri_data["TIMESTAMP_START"], format="%Y%m%d%H%M"
    )
    # Pressure from kPa to Pa
    de_gri_data["PA_F"] = de_gri_data["PA_F"].to_numpy() * 1000
    # PPFD from SWDOWN
    de_gri_data["PPFD"] = de_gri_data["SW_IN_F_MDS"].to_numpy() * 2.04

    return de_gri_data


@pytest.fixture
def de_gri_daily_data():
    """Loads the daily phenology data."""

    # Load the data
    datapath = (
        resources.files("pyrealm_build_data.phenology")
        / "DE_Gri_Grassland_example_subset.csv"
    )

    de_gri_data = pd.read_csv(datapath, na_values=[-9999])

    # Calculate time as np.datetime64 and scale other vars
    de_gri_data["date"] = pd.to_datetime(de_gri_data["date"], format="%Y-%m-%d")
    # Pressure from kPa to Pa
    de_gri_data["PA_F"] = de_gri_data["PA_F"].to_numpy() * 1000

    return de_gri_data
