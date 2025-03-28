"""pytest configuration for the splash submodule."""

from importlib import resources

import pandas as pd
import pytest
import xarray

EXPECTED_NAME_MAP: dict[str, str] = {
    "my_nu": "nu",
    "my_lambda": "lambda_",
    "dr": "distance_factor",
    "delta": "declination",
    "hs": "sunset_hour_angle",
    "ra_d": "daily_solar_radiation",
    "tau": "transmissivity",
    "ppfd_d": "daily_ppfd",
    "rnl": "net_longwave_radiation",
    "hn": "crossover_hour_angle",
    "rn_d": "daytime_net_radiation",
    "rnn_d": "nighttime_net_radiation",
    "sat": "sat",
    "lv": "lv",
    "pw": "pw",
    "psy": "psy",
    "econ": "econ",
    "rx": "rx",
    "hi": "hi",
    "cond": "cond",
    "eet_d": "eet_d",
    "pet_d": "pet_d",
    "aet_d": "aet_d",
    "wn": "wn",
    "ro": "ro",
}


@pytest.fixture
def splash_core_constants():
    """Provide constants using SPLASH original defaults.

    SPLASH v1 uses 15°C / 288.1 5 K in the standard atmosphere definition and uses the
    Chen method for calculating water density.
    """

    from pyrealm.constants import CoreConst

    return CoreConst(k_To=288.15, water_density_method="chen")


@pytest.fixture()
def daily_flux_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test daily values.

    Loads an input file and SPLASH outputs for 100 random locations with a wide range of
    possible input values. Not intended for testing time series iteration, just the
    daily predictions of all core variables.
    """

    dpath = resources.files("pyrealm_build_data.splash.data")
    inputs = pd.read_csv(str(dpath / "daily_flux_benchmark_inputs.csv"))
    expected = pd.read_csv(str(dpath / "daily_flux_benchmark_outputs.csv"))

    # rename fields to match new implementation
    inputs["dates"] = pd.to_datetime(inputs["dates"])
    expected = expected.rename(columns=EXPECTED_NAME_MAP)

    return inputs, expected


@pytest.fixture()
def one_d_benchmark() -> tuple[xarray.Dataset, xarray.Dataset]:
    """Test one dimensional time series.

    Loads the input data and resulting soil moisture outputs from the single location
    San Francisco dataset provided with the original implementation. These were
    originally calculated using the __main__ code in SPLASH main.py, but the data has
    bee converted to netCDF and run using an alternative interface in order to retain
    more validation data.
    """

    dpath = resources.files("pyrealm_build_data.splash")

    inputs = xarray.load_dataset(dpath / "data/splash_sf_example_data.nc")

    expected = xarray.load_dataset(dpath / "data/splash_sf_example_data_details.nc")

    return inputs, expected


@pytest.fixture()
def grid_benchmarks() -> tuple[xarray.Dataset, xarray.Dataset]:
    """Test 3D time series.

    This provides a 20 x 20 cell chunk of daily data from WFDE5 v2 and CRU for the west
    coast of the US over two years. It provides a test of iterated calculations and
    checking that applying functions across arrays, not iterating through individual
    time series gives identical results.
    """

    dpath = resources.files("pyrealm_build_data.splash.data")

    inputs = xarray.load_dataset(dpath / "splash_nw_us_grid_data.nc")

    expected = xarray.load_dataset(dpath / "splash_nw_us_grid_data_outputs.nc")

    return inputs, expected
