import numpy as np
from numpy.testing import assert_array_equal

from pyrealm.canopy_model.functions.jaideep_t_model_extension_functions import *


def test_calculate_q_m_values_returns_q_m_for_valid_input():
    m_values = np.array([2, 3])
    n_values = np.array([5, 4])
    expected_q_m_values = np.array([2.9038988210485766, 2.3953681843215673])
    actual_q_m_values = calculate_q_m_values(m_values, n_values)
    assert_array_equal(actual_q_m_values, expected_q_m_values)


def test_calculate_q_m_values_raises_exception_for_invalid_input():
    pass


def test_calculate_z_m_values():
    pass


def test_calculate_r_0_values():
    pass


def test_calculate_projected_canopy_area_for_individuals():
    pass


def test_calculate_relative_canopy_radii():
    pass