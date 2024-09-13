"""test the functions in t_model_functions.py."""

import numpy as np
from numpy.testing import assert_array_almost_equal

from pyrealm.demography.t_model_functions import (
    calculate_canopy_q_m,
    calculate_crown_areas,
    calculate_crown_fractions,
    calculate_foliage_masses,
    calculate_heights,
    calculate_sapwood_masses,
    calculate_stem_masses,
)


def test_calculate_heights():
    """Tests happy path for calculation of heights of tree from diameter."""
    pft_h_max_values = np.array([25.33, 15.33])
    pft_a_hd_values = np.array([116.0, 116.0])
    diameters_at_breast_height = np.array([0.2, 0.6])
    expected_heights = np.array([15.19414157, 15.16639589])
    actual_heights = calculate_heights(
        h_max=pft_h_max_values,
        a_hd=pft_a_hd_values,
        dbh=diameters_at_breast_height,
    )
    assert_array_almost_equal(actual_heights, expected_heights, decimal=8)


def test_calculate_crown_areas():
    """Tests happy path for calculation of crown areas of trees."""
    pft_ca_ratio_values = np.array([2, 3])
    pft_a_hd_values = np.array([116.0, 116.0])
    diameters_at_breast_height = np.array([0.2, 0.6])
    heights = np.array([15.194142, 15.166396])
    expected_crown_areas = np.array([0.04114983, 0.1848361])
    actual_crown_areas = calculate_crown_areas(
        ca_ratio=pft_ca_ratio_values,
        a_hd=pft_a_hd_values,
        dbh=diameters_at_breast_height,
        height=heights,
    )
    assert_array_almost_equal(actual_crown_areas, expected_crown_areas, decimal=8)


def test_calculate_crown_fractions():
    """Tests happy path for calculation of crown fractions of trees."""
    pft_a_hd_values = np.array([116.0, 116.0])
    diameters_at_breast_height = np.array([0.2, 0.6])
    heights = np.array([15.194142, 15.166396])
    expected_crown_fractions = np.array([0.65491991, 0.21790799])
    actual_crown_fractions = calculate_crown_fractions(
        a_hd=pft_a_hd_values,
        dbh=diameters_at_breast_height,
        height=heights,
    )
    assert_array_almost_equal(
        actual_crown_fractions, expected_crown_fractions, decimal=8
    )


def test_calculate_stem_masses():
    """Tests happy path for calculation of stem masses."""
    diameters_at_breast_height = np.array([0.2, 0.6])
    heights = np.array([15.194142, 15.166396])
    pft_rho_s_values = np.array([200.0, 200.0])
    expected_stem_masses = np.array([47.73380488, 428.8197443])
    actual_stem_masses = calculate_stem_masses(
        dbh=diameters_at_breast_height, height=heights, rho_s=pft_rho_s_values
    )
    assert_array_almost_equal(actual_stem_masses, expected_stem_masses, decimal=8)


def test_calculate_foliage_masses():
    """Tests happy path for calculation of foliage masses."""
    crown_areas = np.array([0.04114983, 0.1848361])
    pft_lai_values = np.array([1.8, 1.8])
    pft_sla_values = np.array([14.0, 14.0])
    expected_foliage_masses = np.array([0.00529069, 0.02376464])
    actual_foliage_masses = calculate_foliage_masses(
        crown_area=crown_areas, lai=pft_lai_values, sla=pft_sla_values
    )
    assert_array_almost_equal(actual_foliage_masses, expected_foliage_masses, decimal=8)


def test_calculate_sapwood_masses():
    """Tests happy path for calculation of sapwood masses."""
    crown_areas = np.array([0.04114983, 0.1848361])
    pft_rho_s_values = np.array([200.0, 200.0])
    heights = np.array([15.194142, 15.166396])
    crown_fractions = np.array([0.65491991, 0.21790799])
    pft_ca_ratio_values = [390.43, 390.43]
    expected_sapwood_masses = np.array([0.21540173, 1.27954667])
    actual_sapwood_masses = calculate_sapwood_masses(
        crown_area=crown_areas,
        rho_s=pft_rho_s_values,
        height=heights,
        crown_fraction=crown_fractions,
        ca_ratio=pft_ca_ratio_values,
    )
    assert_array_almost_equal(actual_sapwood_masses, expected_sapwood_masses, decimal=8)


"""Test the functions in jaideeps_t_model_extension_functions.py."""


def test_calculate_calculate_canopy_q_m_returns_q_m_for_valid_input():
    """Test happy path for calculating q_m.

    test that values of q_m are calculated correctly when valid arguments are
    provided to the function.
    """
    m_values = np.array([2, 3])
    n_values = np.array([5, 4])
    expected_q_m_values = np.array([2.9038988210485766, 2.3953681843215673])
    actual_q_m_values = calculate_canopy_q_m(m=m_values, n=n_values)
    assert_array_almost_equal(actual_q_m_values, expected_q_m_values, decimal=8)


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
