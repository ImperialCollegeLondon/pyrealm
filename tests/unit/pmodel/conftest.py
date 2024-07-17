"""Shared fixture definitions for the PModel unit test suite."""

from importlib import resources

import pandas
import pytest


@pytest.fixture(scope="module")
def be_vie_data():
    """Import the subdaily model benchmark test data."""

    # Load the BE-Vie data
    data_path = (
        resources.files("pyrealm_build_data.subdaily") / "subdaily_BE_Vie_2014.csv"
    )

    data = pandas.read_csv(str(data_path))
    data["time"] = pandas.to_datetime(data["time"])

    return data
