"""Testing jmax_limitation submodule."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def mock_optimal_chi():
    """Build an OptimalChiABC instance."""
    from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14
    from pyrealm.pmodel.pmodel_environment import PModelEnvironment

    # Create an instance of OptimalChiABC for testing
    env = PModelEnvironment(
        tc=np.array([20]),
        vpd=np.array([1000]),
        co2=np.array([400]),
        patm=np.array([101325.0]),
    )
    return OptimalChiPrentice14(env)


@pytest.mark.parametrize(
    "method, classname, expected_f_j, expected_f_v",
    [
        ("none", "JmaxLimitationNone", 1.0, 1.0),
        ("wang17", "JmaxLimitationWang17", 0.66722, 0.55502),
        ("smith19", "JmaxLimitationSmith19", 1.10204, 0.75442),
    ],
)
def test_jmax_limitation(
    mock_optimal_chi, method, classname, expected_f_j, expected_f_v
):
    """Test that JmaxLimitation classes works as expected."""
    from pyrealm.pmodel import jmax_limitation
    from pyrealm.pmodel.jmax_limitation import JMAX_LIMITATION_CLASS_REGISTRY

    # Check the implementation works identically directly and via registry
    for classobj in [
        JMAX_LIMITATION_CLASS_REGISTRY[method],
        getattr(jmax_limitation, classname),
    ]:
        # Create an instance
        jmax_limitation = classobj(optchi=mock_optimal_chi)

        # Assert that calculated f_j and f_v match expected values
        assert_allclose(jmax_limitation.f_j, expected_f_j, rtol=1e-05)
        assert_allclose(jmax_limitation.f_v, expected_f_v, rtol=1e-05)


def test_invalid_method():
    """Test that JmaxLimitation raises ValueError for invalid method."""
    from pyrealm.pmodel.jmax_limitation import JMAX_LIMITATION_CLASS_REGISTRY

    # Test invalid method
    with pytest.raises(KeyError):
        _ = JMAX_LIMITATION_CLASS_REGISTRY["invalid_method"]
