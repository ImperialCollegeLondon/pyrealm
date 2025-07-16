"""Tests the FaparLimitation class."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="aridity_index, raises",
    argvalues=[
        (0, pytest.raises(ValueError)),
        (-1, pytest.raises(ValueError)),
        (1, does_not_raise()),
        (np.array([1, 1, 1]), does_not_raise()),
        (np.array([1, 1, -1]), pytest.raises(ValueError)),
    ],
)
def test_aridity_index_check(aridity_index, raises):
    """Tests if the AI positivity check is works for both scalar and vector inputs."""
    from pyrealm.phenology.fapar_limitation import FaparLimitation

    # Inputs are arrays of length 3
    shape = (3,)
    with raises:
        FaparLimitation(
            annual_total_potential_gpp=np.ones(shape),
            annual_mean_ca=np.ones(shape),
            annual_mean_chi=np.ones(shape),
            annual_mean_vpd=np.ones(shape),
            annual_total_precip=np.ones(shape),
            aridity_index=aridity_index,
        )
