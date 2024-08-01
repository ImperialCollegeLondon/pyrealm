"""Test quantum yield calculations against benchmarks."""

from importlib import resources

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def values():
    """Fixture to load test inputs and expected rpmodel outputs from file."""

    datapath = (
        resources.files("pyrealm_build_data.sandoval_kphio") / "sandoval_kphio.csv"
    )

    with open(str(datapath)) as infile:
        values = pd.read_csv(infile)

    return values


def test_QuantumYieldSandoval(values):
    """Check implementation against values from original R code."""
    from pyrealm.pmodel import PModelEnvironment
    from pyrealm.pmodel.quantum_yield import QuantumYieldSandoval

    env = PModelEnvironment(
        tc=values["temp"].to_numpy(),
        patm=101325,
        vpd=820,
        co2=400,
        mean_growth_temperature=values["mean_gdd_temp"].to_numpy(),
        aridity_index=values["aridity_index"].to_numpy(),
    )

    # Calculate kphio for that environment
    qy = QuantumYieldSandoval(env)

    # Get expected kphio, masking negative values from the reference implementation
    expected = values["phio"].to_numpy()
    expected = np.where(expected < 0, np.nan, expected)

    assert np.allclose(expected, qy.kphio, equal_nan=True)
