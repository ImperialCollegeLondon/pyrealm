"""Test the functions in jaideeps_t_model_extension_functions.py."""

import numpy as np
from numpy.testing import assert_array_equal

from pyrealm.canopy_model.functions.jaideep_t_model_extension_functions import (
    calculate_q_m_values,
)


def test_calculate_q_m_values_returns_q_m_for_valid_input():
    """Test happy path for calculating q_m.

    test that values of q_m are calculated correctly when valid arguments are
    provided to the function.
    """
    m_values = np.array([2, 3])
    n_values = np.array([5, 4])
    expected_q_m_values = np.array([2.9038988210485766, 2.3953681843215673])
    actual_q_m_values = calculate_q_m_values(m_values, n_values)
    assert_array_equal(actual_q_m_values, expected_q_m_values)


def test_calculate_q_m_values_raises_exception_for_invalid_input():
    """Test unhappy path for calculating q_m.

    Test that an exception is raised when invalid arguments are provided to the
    function.
    """

    pass


def test_calculate_z_max_values():
    """Test happy path for calculating z_max."""

    pass


def test_calculate_r_0_values():
    """Test happy path for calculating r_0."""
    pass


def test_calculate_projected_canopy_area_for_individuals():
    """Test happy path for calculating canopy area for individuals."""
    pass


def test_calculate_relative_canopy_radii():
    """Test happy path for calculating relative canopy radii for individuals."""
    pass
