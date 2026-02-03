"""Shared fixtures for phenology regression testing."""

import json
from importlib import resources

import numpy as np
import pandas as pd
import pytest


def dataframe_to_dict_of_nparrays(data):
    """Utility function to preconvert a dataframe of pd.Series to np.array."""

    data_dict = {k: v.to_numpy() for k, v in data.items()}

    return data_dict


@pytest.fixture
def site_data():
    """Load the site data."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.inputs.source")
        / "DE-GRI_site_data.json"
    )

    with open(str(datafile)) as json_src:
        site_data = json.load(json_src)

    return site_data


@pytest.fixture()
def annual_inputs(timescale):
    """Returns the annual input data for one of the two test timescales."""

    ts_directory = "subdaily" if timescale == "hh" else "fortnightly"
    datafile = (
        resources.files(f"pyrealm_build_data.phenology.inputs.{ts_directory}")
        / "annual_inputs.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def daily_assimilation(timescale):
    """Returns the daily assimilation data for one of the two test timescales."""

    ts_directory = "subdaily" if timescale == "hh" else "fortnightly"
    datafile = (
        resources.files(f"pyrealm_build_data.phenology.inputs.{ts_directory}")
        / "daily_assimilation.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def pmodel_inputs(timescale):
    """Returns the pmodel input data for one of the two test timescales."""

    ts_directory = "subdaily" if timescale == "hh" else "fortnightly"
    datafile = (
        resources.files(f"pyrealm_build_data.phenology.inputs.{ts_directory}")
        / "pmodel_inputs.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def pmodel_outputs(timescale):
    """Returns the pmodel output data for one of the two test timescales."""

    ts_directory = "subdaily" if timescale == "hh" else "fortnightly"
    datafile = (
        resources.files(f"pyrealm_build_data.phenology.inputs.{ts_directory}")
        / "pmodel_outputs.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def fapar_max_predictions(method_predictions_dir):
    """Returns the annual fapar_max predictions for a given method."""

    datafile = (
        resources.files(f"pyrealm_build_data.phenology.{method_predictions_dir}")
        / "fapar_max_predictions.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def daily_lai_predictions(method_predictions_dir):
    """Returns the annual fapar_max predictions for a given method."""

    datafile = (
        resources.files(f"pyrealm_build_data.phenology.{method_predictions_dir}")
        / "daily_lai_predictions.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def data_fapar_limitation(timescale, method_predictions_dir):
    """Load the fapar limitation input data from csv file."""

    inputs = pd.read_csv(
        str(
            resources.files(f"pyrealm_build_data.phenology.inputs.{timescale}")
            / "annual_inputs.csv"
        )
    )

    predictions = pd.read_csv(
        str(
            resources.files(f"pyrealm_build_data.phenology.{method_predictions_dir}")
            / "fapar_max_predictions.csv"
        )
    )

    data = inputs.merge(predictions)
    return dataframe_to_dict_of_nparrays(data)


@pytest.fixture()
def data_phenology(timescale, method_predictions_dir):
    """Load the fapar limitation input data from csv file."""

    inputs = pd.read_csv(
        str(
            resources.files(f"pyrealm_build_data.phenology.inputs.{timescale}")
            / "daily_assimilation.csv"
        )
    )

    predictions = pd.read_csv(
        str(
            resources.files(f"pyrealm_build_data.phenology.{method_predictions_dir}")
            / "daily_lai_predictions.csv"
        )
    )

    data = inputs.merge(predictions)
    return dataframe_to_dict_of_nparrays(data)


@pytest.fixture
def phenology_pmodels(
    pmodel_inputs,
    timescale,  # Also parameterises pmodel_inputs
    pmodel_year,
):
    """Create test PModel inputs.

    Returns PModels, datetimes and penalty factors for the fortnightly and subdaily
    datasets, for use in test from_pmodel methods.

    To support testing of the Zhu method via the from_pmodel method, the fixture can be
    parameterised to return all years (pmodel_year=None) or a single year.
    """
    from pyrealm.pmodel import (
        AcclimationModel,
        PModel,
        PModelEnvironment,
        SubdailyPModel,
    )

    if pmodel_year is not None:
        subset = (
            pmodel_inputs["time"].astype("datetime64[Y]").astype(str).astype(int)
            == pmodel_year
        )
    else:
        subset = np.ones_like(pmodel_inputs["time"], dtype=np.bool_)

    env = PModelEnvironment(
        tc=pmodel_inputs["tc"][subset],
        vpd=pmodel_inputs["vpd"][subset],
        co2=pmodel_inputs["co2"][subset],
        patm=pmodel_inputs["patm"][subset],
        fapar=pmodel_inputs["fapar"][subset],
        ppfd=pmodel_inputs["ppfd"][subset],
    )

    pmodel_inputs["time"] = pmodel_inputs["time"].astype("datetime64[s]")

    # The two timescales use different PModels and also need to specify the datetimes
    # and gpp penalty factors in FaparLimitation differently
    if timescale == "ft":
        # Fit PModel
        pmodel = PModel(
            env=env,
            reference_kphio=1 / 8,
            method_kphio="temperature",
        )
        # Define datetimes of observations - no GPP penalty
        fl_datetimes = pmodel_inputs["time"][subset]
        gpp_penalty_factor = None

    else:
        # Set up the datetimes of the observations and set the acclimation window
        acclim = AcclimationModel(
            datetimes=pmodel_inputs["time"][subset],
            alpha=1 / 15,
        )
        acclim.set_window(
            window_center=np.timedelta64(12, "h"),
            half_width=np.timedelta64(30, "m"),
        )

        # Fit the subdaily PModel
        pmodel = SubdailyPModel(
            env=env,
            acclim_model=acclim,
            reference_kphio=1 / 8,
            method_kphio="temperature",
        )

        # FaparLimitation uses the datetimes from pmodel.acclim_model and uses a soil
        # moisture stress penalty
        fl_datetimes = None
        gpp_penalty_factor = pmodel_inputs["soilm_stress"][subset]

    return pmodel, fl_datetimes, gpp_penalty_factor
