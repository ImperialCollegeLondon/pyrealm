"""This module contains regression tests for the two leaf model code."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pyrealm.pmodel import PModel
from pyrealm.pmodel.pmodel_environment import PModelEnvironment
from pyrealm.pmodel.two_leaf import TwoLeafAssimilation, TwoLeafIrradiance


@pytest.fixture
def get_data():
    """Reads in data to assess tests against."""
    forcing_data = pd.read_csv("pyrealm_build_data/two_leaf/merged_BE-Vie_data.csv")
    forcing_data["time"] = pd.to_datetime(forcing_data["time"])

    return forcing_data


@pytest.fixture
def solar_elevation():
    """Defines solar elevation value derived from R code supplied by D. Orme."""
    return np.array([0.98212117])


@pytest.fixture
def solar_irradiance(solar_elevation, get_data):
    """Creates instace of TwoLeafIrradiance class."""

    data = get_data.loc[(get_data["time"] == "2014-08-01 12:30:00")]

    PPFD = data["ppfd"].to_numpy()
    LAI = data["LAI"].to_numpy()
    PATM = data["patm"].to_numpy()
    result = TwoLeafIrradiance(solar_elevation, PPFD, LAI, PATM)

    return result


@pytest.fixture
def assimilation_single_day(solar_irradiance, get_data):
    """Tests TwoLeafAssimilation gpp_estimator method against reference data."""

    data = get_data.loc[(get_data["time"] == "2014-08-01 12:30:00")]

    pmod_env = PModelEnvironment(
        tc=data["tc"].to_numpy(),
        mean_growth_temperature=data["tc"].to_numpy(),
        patm=data["patm"].to_numpy(),
        co2=data["co2"].to_numpy(),
        vpd=data["vpd"].to_numpy(),
        fapar=data["fapar"].to_numpy(),
        ppfd=data["ppfd"].to_numpy(),
    )

    # Standard P Model, simple Arrhenius scaling, phi_0 = 1/8
    standard_pmod = PModel(pmod_env)

    assim = TwoLeafAssimilation(
        pmodel=standard_pmod,
        irradiance=solar_irradiance,
    )

    return assim


def test_two_leaf_irradiance(solar_irradiance, get_data):
    """Tests calc_absorbed_irradiance method."""

    test_values = {
        "diffuse_irradiance": np.array([136.29450038]),
        "beam_irradiance": np.array([714.70549962]),
        "beam_extinction_coef": np.array([0.6011949063494533]),
        "scattered_beam_extinction_coef": np.array([0.55309931]),
        "beam_reflectance": np.array([0.03003319]),
        "sunlit_beam_irradiance": np.array([485.32862357]),
        "sunlit_diffuse_irradiance": np.array([69.44255865]),
        "sunlit_scattered_irradiance": np.array([25.43894747]),
        "sunlit_absorbed_irradiance": np.array([580.2101296936461]),
        "canopy_irradiance": np.array([646.830682]),
        "shaded_absorbed_irradiance": np.array([66.620553]),
    }

    for attr, expected_value in test_values.items():
        value = getattr(solar_irradiance, attr, None)
        assert value is not None
        assert_allclose(value, expected_value)


def test_two_leaf_assimilation(assimilation_single_day):
    """Tests TwoLeafAssimilation class against reference data.

    TODO - this reference data is currently circular. I'm not sure we have a sensible
    golden dataset to validate against.
    """

    test_values = {
        "canopy_extinction_coef": np.array([0.17377901]),
        "Vcmax25_canopy": np.array([220.02686435]),
        "Vcmax25_sun": np.array([112.04497916]),
        "Vcmax25_shade": np.array([107.98188519]),
        "Jmax25_sun": np.array([212.85376583]),
        "Jmax25_shade": np.array([206.19029171]),
        "Vcmax_sun": np.array([88.0640465]),
        "Vcmax_shade": np.array([84.87057456]),
        "Jmax_sun": np.array([181.04942389]),
        "Jmax_shade": np.array([175.38159769]),
        "Av_sun": np.array([23.67492383]),
        "Av_shade": np.array([22.81639861]),
        "J_sun": np.array([91.24985393]),
        "J_shade": np.array([21.94981797]),
        "Aj_sun": np.array([15.24637566]),
        "Aj_shade": np.array([3.66745979]),
        "Acanopy_sun": np.array([15.24637566]),
        "Acanopy_shade": np.array([3.66745979]),
        "gpp": np.array([227.16840351]),
    }

    for attr, expected_value in test_values.items():
        value = getattr(assimilation_single_day, attr, None)
        assert value is not None
        assert_allclose(value, expected_value)
