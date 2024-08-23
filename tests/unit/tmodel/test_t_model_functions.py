"""test the functions in t_model_functions.py."""

import numpy as np
from numpy.testing import assert_array_almost_equal

from pyrealm.t_model_functions import (
    calculate_crown_areas,
    calculate_crown_fractions,
    calculate_foliage_masses,
    calculate_heights,
    calculate_stem_masses,
    calculate_swd_masses,
)


def test_calculate_heights():
    """Tests happy path for calculation of heights of tree from diameter."""
    pft_h_max_values = np.array([25.33, 15.33])
    pft_a_hd_values = np.array([116.0, 116.0])
    diameters_at_breast_height = np.array([0.2, 0.6])
    expected_heights = np.array([15.19414157, 15.16639589])
    actual_heights = calculate_heights(
        pft_h_max_values, pft_a_hd_values, diameters_at_breast_height
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
        pft_ca_ratio_values, pft_a_hd_values, diameters_at_breast_height, heights
    )
    assert_array_almost_equal(actual_crown_areas, expected_crown_areas, decimal=8)


def test_calculate_crown_fractions():
    """Tests happy path for calculation of crown fractions of trees."""
    pft_a_hd_values = np.array([116.0, 116.0])
    diameters_at_breast_height = np.array([0.2, 0.6])
    heights = np.array([15.194142, 15.166396])
    expected_crown_fractions = np.array([0.65491991, 0.21790799])
    actual_crown_fractions = calculate_crown_fractions(
        heights, pft_a_hd_values, diameters_at_breast_height
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
        diameters_at_breast_height, heights, pft_rho_s_values
    )
    assert_array_almost_equal(actual_stem_masses, expected_stem_masses, decimal=8)


def test_calculate_foliage_masses():
    """Tests happy path for calculation of foliage masses."""
    crown_areas = np.array([0.04114983, 0.1848361])
    pft_lai_values = np.array([1.8, 1.8])
    pft_sla_values = np.array([14.0, 14.0])
    expected_foliage_masses = np.array([0.00529069, 0.02376464])
    actual_foliage_masses = calculate_foliage_masses(
        crown_areas, pft_lai_values, pft_sla_values
    )
    assert_array_almost_equal(actual_foliage_masses, expected_foliage_masses, decimal=8)


def test_calculate_swd_masses():
    """Tests happy path for calculation of ??? masses."""
    crown_areas = np.array([0.04114983, 0.1848361])
    pft_rho_s_values = np.array([200.0, 200.0])
    heights = np.array([15.194142, 15.166396])
    crown_fractions = np.array([0.65491991, 0.21790799])
    pft_ca_ratio_values = [390.43, 390.43]
    expected_swd_masses = np.array([0.21540173, 1.27954667])
    actual_swd_masses = calculate_swd_masses(
        crown_areas, pft_rho_s_values, heights, crown_fractions, pft_ca_ratio_values
    )
    assert_array_almost_equal(actual_swd_masses, expected_swd_masses, decimal=8)
