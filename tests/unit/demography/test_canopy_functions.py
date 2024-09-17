"""test the functions in canopy_functions.py."""

import numpy as np


def test_calculate_calculate_canopy_q_m_returns_q_m_for_valid_input():
    """Test happy path for calculating q_m.

    test that values of q_m are calculated correctly when valid arguments are
    provided to the function.
    """

    from pyrealm.demography.canopy_functions import calculate_canopy_q_m

    m_values = np.array([2, 3])
    n_values = np.array([5, 4])
    expected_q_m_values = np.array([2.9038988210485766, 2.3953681843215673])
    actual_q_m_values = calculate_canopy_q_m(m=m_values, n=n_values)

    assert np.allclose(actual_q_m_values, expected_q_m_values)


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


def test_calculate_total_community_crown_area():
    """Test happy path for calculating total community crown area."""
    pass


def test_calculate_number_of_canopy_layers():
    """Test happy path for calculating number of canopy layers."""
    pass


def calculate_community_projected_area_at_z():
    """Test happy path for calculating total projected area for community."""
    pass


def test_calculate_canopy_layer_heights():
    """Test happy path for calculation of canopy layer heights."""
    pass


def test_solve_canopy_closure_height():
    """Test happy path for solver function for canopy closure height."""
    pass


def test_calculate_projected_leaf_area_for_individuals():
    """Test happy path for calculating projected leaf area for individuals."""
    pass


def test_calculate_total_canopy_A_cp():
    """Test happy path for calculating total canopy A_cp across a community."""
    pass
