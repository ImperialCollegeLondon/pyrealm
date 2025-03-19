"""This module contains regression tests for the two leaf model code."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pyrealm.pmodel import PModel
from pyrealm.pmodel.pmodel_environment import PModelEnvironment
from pyrealm.pmodel.two_leaf import TwoLeafAssimilation, TwoLeafIrradience


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
    """Creates instace of TwoLeafIrradience class."""

    data = get_data.loc[(get_data["time"] == "2014-08-01 12:30:00")]

    PPFD = data["ppfd"].to_numpy()
    LAI = data["LAI"].to_numpy()
    PATM = data["patm"].to_numpy()
    result = TwoLeafIrradience(solar_elevation, PPFD, LAI, PATM)

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

    # Standard P Model
    standard_pmod = PModel(
        pmod_env, method_arrhenius="kattge_knorr", reference_kphio=0.081785
    )

    solar_irradiance.calc_absorbed_irradiance()

    assim = TwoLeafAssimilation(
        pmodel=standard_pmod,
        irrad=solar_irradiance,
    )

    assim.gpp_estimator()

    return assim


def test_two_leaf_irradiance(solar_irradiance, get_data):
    """Tests calc_absorbed_irradiance method."""
    solar_irradiance.calc_absorbed_irradiance()

    test_values = (
        ("beam_extinction_coefficient", np.array([0.6011949063494533])),
        ("scattered_beam_extinction_coefficient", np.array([0.55309931])),
        ("horizontal_leaf_beam_irradiance", np.array([0.04060739027615024])),
        ("uniform_leaf_beam_irradiance", np.array([0.03003319])),
        ("diffuse_irradiance", np.array([136.29450038])),
        ("beam_irradiance", np.array([714.70549962])),
        ("canopy_irradiance", np.array([636.08707023])),
        ("sunlit_beam_irradiance", np.array([485.32862357])),
        ("sunlit_diffuse_irradiance", np.array([69.44255865])),
        ("sunlit_scattered_irradiance", np.array([25.43894747])),
        ("shaded_absorbed_irradiance", np.array([55.876940533221614])),
        ("sunlit_absorbed_irradiance", np.array([580.2101296936461])),
    )

    for attr, expected_value in test_values:
        value = getattr(solar_irradiance, attr, None)
        assert value is not None
        assert_allclose(value, expected_value)


def test_two_leaf_assimilation(assimilation_single_day):
    """Tests TwoLeafAssimilation class against reference data."""
    assimilation = assimilation_single_day
    assert np.allclose(assimilation.Vmax25_canopy, np.array([146.73080481]))
    assert np.allclose(assimilation.Vmax25_sun, np.array([74.37653372]))
    assert np.allclose(assimilation.Vmax25_shade, np.array([72.35427109]))
    assert np.allclose(assimilation.Vmax_sun, np.array([58.55709373]))
    assert np.allclose(assimilation.Vmax_shade, np.array([56.96495416]))
    assert np.allclose(assimilation.Av_sun, np.array([15.74234649]))
    assert np.allclose(assimilation.Av_shade, np.array([15.31431957]))
    assert np.allclose(assimilation.Jmax25_sun, np.array([151.0775153]))
    assert np.allclose(assimilation.Jmax25_shade, np.array([147.76100459]))
    assert np.allclose(assimilation.Jmax_sun, np.array([128.43874966]))
    assert np.allclose(assimilation.Jmax_shade, np.array([125.61921369]))
    assert np.allclose(assimilation.J_sun, np.array([73.41800149]))
    assert np.allclose(assimilation.J_shade, np.array([17.9579488]))
    assert np.allclose(assimilation.Aj_sun, np.array([12.26696135]))
    assert np.allclose(assimilation.Aj_shade, np.array([3.00048298]))
    assert np.allclose(assimilation.Acanopy_sun, np.array([12.26696135]))
    assert np.allclose(assimilation.Acanopy_shade, np.array([3.00048298]))
    assert np.allclose(assimilation.gpp_estimate, np.array([150.33527564]))
