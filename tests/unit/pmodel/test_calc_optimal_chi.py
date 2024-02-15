"""Testing CalcOptimal submodule."""


import numpy as np
import pytest

from pyrealm.pmodel.optimal_chi import (
    OptimalChiC4,
    OptimalChiC4NoGamma,
    OptimalChiC4RootzoneStress,
    OptimalChiLavergne20C3,
    OptimalChiLavergne20C4,
    OptimalChiPrentice14,
    OptimalChiPrentice14RootzoneStress,
)
from pyrealm.pmodel.pmodel_environment import PModelEnvironment


@pytest.fixture
def photo_env():
    """Photosynthesis Environment setup."""
    return PModelEnvironment(
        tc=np.array([20]),
        vpd=np.array([1000]),
        co2=np.array([400]),
        patm=np.array([101325.0]),
        rootzonestress=np.array([1]),
        theta=np.array([0.5]),
    )


@pytest.mark.parametrize(
    """optimal_chi_class""",
    [
        OptimalChiPrentice14,
        OptimalChiPrentice14RootzoneStress,
        OptimalChiC4,
        OptimalChiC4RootzoneStress,
        OptimalChiLavergne20C3,
        OptimalChiLavergne20C4,
        OptimalChiC4NoGamma,
    ],
)
def test_set_beta(optimal_chi_class, photo_env):
    """Test that beta is set correctly."""
    optimal_chi_instance = optimal_chi_class(env=photo_env)
    optimal_chi_instance.set_beta()
    # Test that beta attribute is set correctly
    assert isinstance(optimal_chi_instance.beta, np.ndarray)


@pytest.mark.parametrize(
    """optimal_chi_class""",
    [
        OptimalChiPrentice14,
        OptimalChiPrentice14RootzoneStress,
        OptimalChiC4,
        OptimalChiC4RootzoneStress,
        OptimalChiLavergne20C3,
        OptimalChiLavergne20C4,
        OptimalChiC4NoGamma,
    ],
)
def test_estimate_chi(optimal_chi_class, photo_env):
    """Test that chi is estimated correctly."""
    optimal_chi_instance = optimal_chi_class(env=photo_env)
    optimal_chi_instance.set_beta()
    optimal_chi_instance.estimate_chi()
    # Test that chi and other related attributes are calculated correctly
    assert isinstance(optimal_chi_instance.chi, np.ndarray)
    assert isinstance(optimal_chi_instance.mc, np.ndarray)
    assert isinstance(optimal_chi_instance.mj, np.ndarray)
    assert isinstance(optimal_chi_instance.mjoc, np.ndarray)
