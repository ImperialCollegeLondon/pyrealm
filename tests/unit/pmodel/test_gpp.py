"""Test that daily GPP is correctly computed for PModel and Subdaily PModel."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyrealm.pmodel import PModel


@pytest.fixture
def test_pmodel():
    """Parameters to >daily to daily gpp test."""
    from pyrealm.pmodel import PModelEnvironment

    env = PModelEnvironment(
        tc=np.array([20.0]),
        theta=np.array([0.5]),
        patm=np.array([101325]),
        vpd=np.array([100]),
        co2=np.array([400]),
    )
    pmodel = PModel(env, method_optchi="lavergne20_c4")

    return pmodel


@pytest.fixture
def test_subdailypmodel(datetimes):
    """Parameters to subdaily to daily gpp test."""
    from pyrealm.pmodel import PModelEnvironment
    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.pmodel import SubdailyPModel

    # Calculate the PModel photosynthetic environment
    env = PModelEnvironment(
        tc=np.array([20]),
        vpd=np.array([2000]),
        co2=np.array([400]),
        patm=np.array([101325.0]),
        fapar=np.array([1]),
        ppfd=np.array([800]),
    )

    # Set up the datetimes of the observations and set the acclimation window
    acclim_model = AcclimationModel(datetimes=np.array(datetimes).ravel())
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Fit the potential GPP: fAPAR = 1 and phi0 = 1/8
    subdaily_pmodel = SubdailyPModel(
        env=env,
        acclim_model=acclim_model,
        reference_kphio=1 / 8,
    )

    return subdaily_pmodel


@pytest.mark.parametrize(
    argnames="datetimes, gpp_in, expected_gpp_out",
    argvalues=[
        pytest.param(
            [
                np.datetime64("2022-01-01"),
                np.datetime64("2022-01-03"),
                np.datetime64("2022-01-05"),
                np.datetime64("2022-01-07"),
                np.datetime64("2022-01-09"),
                np.datetime64("2022-01-11"),
            ],
            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        )
    ],
)
def test_pmodel_get_daily_gpp(datetimes, gpp_in, expected_gpp_out, test_pmodel):
    """Tests that the interpolation to daily gpp from PModel gpp works correctly."""

    test_pmodel.gpp = gpp_in
    daily_gpp = test_pmodel._get_daily_gpp(np.array(datetimes))
    assert_allclose(expected_gpp_out, daily_gpp)


@pytest.mark.parametrize(
    argnames="datetimes, gpp_in, expected_gpp_out",
    argvalues=[
        pytest.param(
            [
                np.arange(
                    np.datetime64("2011-01-01"),
                    np.datetime64("2011-01-31"),
                    np.timedelta64(6, "h"),
                )
            ],
            np.ones(4 * 31),
            np.ones(31),
        )
    ],
)
def test_subdailypmodel_get_daily_gpp(gpp_in, expected_gpp_out, test_subdailypmodel):
    """Tests that the averaging from subdaily gpp to daily gpp works correctly."""

    test_subdailypmodel.gpp = gpp_in
    test_subdailypmodel.acclim_model.n_days = 31
    test_subdailypmodel.acclim_model.n_obs = 4
    assert_allclose(test_subdailypmodel._get_daily_gpp(), expected_gpp_out)
