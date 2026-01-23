"""Shared fixtures for phenology regression testing."""

import json
from importlib import resources

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
def fapar_max_predictions(phenology_method):
    """Returns the annual fapar_max predictions for a given method."""

    datafile = (
        resources.files(f"pyrealm_build_data.phenology.{phenology_method}")
        / "fapar_max_predictions.csv"
    )

    return dataframe_to_dict_of_nparrays(pd.read_csv(str(datafile)))


@pytest.fixture()
def daily_lai_predictions(phenology_method):
    """Returns the annual fapar_max predictions for a given method."""

    datafile = (
        resources.files(f"pyrealm_build_data.phenology.{phenology_method}")
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
            / "daily_inputs.csv"
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
