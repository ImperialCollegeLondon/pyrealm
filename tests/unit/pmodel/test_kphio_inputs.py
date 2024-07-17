"""Test kphio inputs to PModel.

These tests check that the behaviour of the array kphio input options work as
expected. The input values here are the simple scalar inputs to the rpmodel regression
tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def basic_inputs_and_expected():
    """Provides some simple test values and expectations."""
    return (
        SimpleNamespace(
            tc=np.array([20]),
            patm=np.array([101325]),
            co2=np.array([400]),
            vpd=np.array([1000]),
            fapar=np.array([1]),
            ppfd=np.array([300]),
        ),
        SimpleNamespace(
            gpp=np.array([71.2246803]),
            chi=np.array([0.69435201]),
            vcmax=np.array([17.75034477]),
            jmax=np.array([40.03293277]),
        ),
    )


def test_scalar_kphio(basic_inputs_and_expected):
    """Test that the basic test inputs give the correct answer."""

    from pyrealm.pmodel import PModel, PModelEnvironment

    inputs, expected = basic_inputs_and_expected

    env = PModelEnvironment(
        tc=inputs.tc, patm=inputs.patm, co2=inputs.co2, vpd=inputs.vpd
    )
    mod = PModel(env, kphio=0.05, do_ftemp_kphio=False)
    mod.estimate_productivity(fapar=inputs.fapar, ppfd=inputs.ppfd)

    assert np.allclose(mod.gpp, expected.gpp)
    assert np.allclose(mod.optchi.chi, expected.chi)
    assert np.allclose(mod.jmax, expected.jmax)
    assert np.allclose(mod.vcmax, expected.vcmax)


@pytest.fixture
def variable_kphio(self, basic_inputs_and_expected):
    """Calculates gpp with alternate kphio by iteration.

    This uses the original single scalar value approach (tested above) to get
    expected values for a wide range of kphio values by iteration.
    """

    from pyrealm.pmodel import PModel, PModelEnvironment

    inputs, _ = basic_inputs_and_expected

    # Set a large range of values to use in testing
    kphio_values = np.arange(0.001, 0.126, step=0.001)
    gpp = np.empty_like(kphio_values)

    for idx, kph in enumerate(kphio_values):
        env = PModelEnvironment(
            tc=inputs.tc, patm=inputs.patm, co2=inputs.co2, vpd=inputs.vpd
        )
        mod = PModel(env, kphio=kph, do_ftemp_kphio=False)
        mod.estimate_productivity(fapar=inputs.fapar, ppfd=inputs.ppfd)
        gpp[idx] = mod.gpp

    return kphio_values, gpp


@pytest.mark.parametrize(argnames="shape", argvalues=[(125,), (25, 5), (5, 5, 5)])
def test_kphio_arrays(self, basic_inputs_and_expected, variable_kphio, shape):
    """Check behaviour with array inputs of kphio."""

    from pyrealm.pmodel import PModel, PModelEnvironment

    inputs, _ = basic_inputs_and_expected
    kphio_vals, expected_gpp = variable_kphio

    env = PModelEnvironment(
        tc=np.broadcast_to(inputs.tc, shape),
        patm=np.broadcast_to(inputs.patm, shape),
        co2=np.broadcast_to(inputs.co2, shape),
        vpd=np.broadcast_to(inputs.vpd, shape),
    )
    mod = PModel(env, kphio=kphio_vals.reshape(shape), do_ftemp_kphio=False)
    mod.estimate_productivity(fapar=inputs.fapar, ppfd=inputs.ppfd)

    assert np.allclose(mod.gpp, expected_gpp.reshape(shape))
