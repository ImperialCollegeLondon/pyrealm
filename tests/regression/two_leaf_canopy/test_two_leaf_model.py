"""TBC."""

import numpy as np
import pandas as pd
import pytest

from pyrealm.pmodel.two_leaf_irradience import TwoLeafIrradience


@pytest.fixture
def get_data():
    """Docstring."""
    forcing_data = pd.read_csv("pyrealm_build_data/two_leaf/merged_BE-Vie_data.csv")
    forcing_data["time"] = pd.to_datetime(forcing_data["time"])

    return forcing_data


@pytest.fixture
def solar_elevation():
    """Docstring."""
    # from Keiths code
    return np.array([0.982121])


def test_two_leaf_irradience(solar_elevation, get_data):
    """Docstring."""
    # extract values at `2014-08-01 12:30:00`
    data = get_data.loc[(get_data["time"] == "2014-08-01 12:30:00")]

    PPFD = (data["ppfd"].to_numpy(),)
    LAI = (data["LAI"].to_numpy(),)
    PATM = data["patm"].to_numpy()
    result = TwoLeafIrradience(solar_elevation, PPFD, LAI, PATM)
    result.calc_absorbed_irradience()

    assert np.allclose(result.kb, np.array([0.6011949063494533]))
    assert np.allclose(result.I_cshade, np.array([55.876940533221614]))
    assert np.allclose(result.I_csun, np.array([580.2101296936461]))
