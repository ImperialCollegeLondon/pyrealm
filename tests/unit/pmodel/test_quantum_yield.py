"""Test the quantum yield implementations."""

import numpy as np
import pytest


@pytest.fixture
def quantum_yield_env():
    """Simple fixture for providing the quantum yield inputs."""
    from pyrealm.pmodel import PModelEnvironment

    return PModelEnvironment(
        tc=np.array([5, 10, 15, 20, 25, 30]),
        patm=101325,
        vpd=820,
        co2=400,
        mean_growth_temperature=np.array([10, 20, 10, 20, 10, 20]),
        aridity_index=np.array([0.9, 0.9, 2, 2, 5, 5]),
    )


@pytest.mark.parametrize(
    argnames="reference_kphio, expected_kphio",
    argvalues=(
        pytest.param(None, 0.049977, id="default"),
        pytest.param(0.1, 0.1, id="provided"),
    ),
)
def test_QuantumYieldConstant(quantum_yield_env, reference_kphio, expected_kphio):
    """Test the constant method."""

    from pyrealm.pmodel.quantum_yield import QuantumYieldConstant

    qy = QuantumYieldConstant(
        env=quantum_yield_env,
        reference_kphio=reference_kphio,
    )

    # Should be a scalar
    assert np.allclose(qy.kphio, np.array([expected_kphio]))


@pytest.mark.parametrize(
    argnames="reference_kphio, expected_kphio_factor",
    argvalues=(
        pytest.param(
            None, np.array([0.4535, 0.538, 0.6055, 0.656, 0.6895, 0.706]), id="default"
        ),
        pytest.param(
            0.1, np.array([0.4535, 0.538, 0.6055, 0.656, 0.6895, 0.706]), id="provided"
        ),
    ),
)
def test_QuantumYieldBernacchiC3(
    quantum_yield_env, reference_kphio, expected_kphio_factor
):
    """Test the Bernacchi temperature method for C3 plants."""

    from pyrealm.pmodel.quantum_yield import QuantumYieldBernacchiC3

    qy = QuantumYieldBernacchiC3(
        env=quantum_yield_env,
        reference_kphio=reference_kphio,
    )

    # The expected_kphio_factor values are the output of the previous implementation
    # (calc_ftemp_kphio), which returned the temperature factors that then needed
    # multiplying by the reference kphio.
    assert np.allclose(qy.kphio, qy.reference_kphio * expected_kphio_factor)


@pytest.mark.parametrize(
    argnames="reference_kphio, expected_kphio_factor",
    argvalues=(
        pytest.param(
            None,
            np.array([0.0744, 0.1896, 0.2816, 0.3504, 0.396, 0.4184]),
            id="default",
        ),
        pytest.param(
            0.1,
            np.array([0.0744, 0.1896, 0.2816, 0.3504, 0.396, 0.4184]),
            id="provided",
        ),
    ),
)
def test_QuantumYieldBernacchiC4(
    quantum_yield_env, reference_kphio, expected_kphio_factor
):
    """Test the Bernacchi temperature method for C4 plants."""

    from pyrealm.pmodel.quantum_yield import QuantumYieldBernacchiC4

    qy = QuantumYieldBernacchiC4(
        env=quantum_yield_env,
        reference_kphio=reference_kphio,
    )

    # The expected_kphio_factor values are the output of the previous implementation
    # (calc_ftemp_kphio), which returned the temperature factors that then needed
    # multiplying by the reference kphio.
    assert np.allclose(qy.kphio, qy.reference_kphio * expected_kphio_factor)


@pytest.mark.parametrize(
    argnames="reference_kphio, expected_kphio",
    argvalues=(
        pytest.param(
            None,
            np.array(
                [0.02848466, 0.05510828, 0.06099888, 0.07537036, 0.02231382, 0.03026185]
            ),
            id="default",
        ),
        pytest.param(
            1 / 8,
            np.array(
                [0.03204524, 0.06199681, 0.06862374, 0.08479165, 0.02510305, 0.03404458]
            ),
            id="provided",
        ),
    ),
)
def test_QuantumYieldSandoval(quantum_yield_env, reference_kphio, expected_kphio):
    """Test the Sandoval temperature and aridity method.

    The test values here have been calculated using the original R implementation
    provided in pyrealm_build_data/sandoval_kphio/calc_phi0.R. A more complete check
    across a wider range of values is provided in the regression tests but this also
    tests the provisision of alternative reference_kphio.

    > source('calc_phi0.R')
    > tc <- c(5, 10, 15, 20, 25, 30)
    > ai <- c(0.9, 0.9, 2, 2, 5, 5)
    > gdd0 <-  c(10, 20, 10, 20, 10, 20)
    > round(mapply(calc_phi0, ai, tc, gdd0), 5)
    > round(mapply(calc_phi0, ai, tc, gdd0, MoreArgs=list(phi_o_theo=1/8)), 5)
    """

    from pyrealm.pmodel.quantum_yield import QuantumYieldSandoval

    qy = QuantumYieldSandoval(
        env=quantum_yield_env,
        reference_kphio=reference_kphio,
    )

    assert np.allclose(qy.kphio, expected_kphio)
