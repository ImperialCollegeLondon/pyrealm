import pytest
import numpy as np
from numpy.testing import assert_array_equal

from pyrealm.canopy_model.model.jaideep_t_model_extension import *


def test_calculate_q_m_values_returns_q_m_for_valid_input():
    m_values = np.array()
    n_values = np.array()
    expected_q_m_values = np.array()
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



