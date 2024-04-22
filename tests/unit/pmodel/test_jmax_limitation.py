"""Testing jmax_limitation submodule."""

import numpy as np
import pytest

# from pyrealm.constants import PModelConst
from pyrealm.pmodel.jmax_limitation import JmaxLimitation
from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14
from pyrealm.pmodel.pmodel_environment import PModelEnvironment


@pytest.fixture
def mock_optimal_chi():
    """Build an OptimalChiABC instance."""
    # Create an instance of OptimalChiABC for testing
    env = PModelEnvironment(
        tc=np.array([20]),
        vpd=np.array([1000]),
        co2=np.array([400]),
        patm=np.array([101325.0]),
    )
    return OptimalChiPrentice14(env)


@pytest.mark.parametrize(
    "method, expected_f_j, expected_f_v",
    [
        ("simple", [1.0], [1.0]),
        (
            "wang17",
            pytest.approx([0.66722], rel=1e-3),
            pytest.approx([0.55502], rel=1e-3),
        ),
        (
            "smith19",
            pytest.approx([1.10204], rel=1e-3),
            pytest.approx([0.75442], rel=1e-3),
        ),
    ],
)
def test_jmax_limitation(mock_optimal_chi, method, expected_f_j, expected_f_v):
    """Test that JmaxLimitation class works as expected."""
    # Create an instance of JmaxLimitation with mock OptimalChiABC and test method
    jmax_limitation = JmaxLimitation(mock_optimal_chi, method=method)
    # Assert that calculated f_j and f_v match expected values
    assert jmax_limitation.f_j == expected_f_j
    assert jmax_limitation.f_v == expected_f_v


def test_invalid_method(mock_optimal_chi):
    """Test that JmaxLimitation raises ValueError for invalid method."""
    # Test invalid method
    with pytest.raises(ValueError):
        JmaxLimitation(mock_optimal_chi, method="invalid_method")
