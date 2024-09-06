"""Populate t model arrays.

Populate the relevant initialised arrays with properties calculated using the
t model.

"""

import numpy as np
from numpy.typing import NDArray


def calculate_heights(
    pft_h_max_values: NDArray[np.float32],
    pft_a_hd_values: NDArray[np.float32],
    diameters_at_breast_height: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Height of tree from diameter, Equation (4) of Li ea."""
    heights = pft_h_max_values * (
        1 - np.exp(-pft_a_hd_values * diameters_at_breast_height / pft_h_max_values)
    )

    return heights


def calculate_crown_areas(
    pft_ca_ratio_values: NDArray[np.float32],
    pft_a_hd_values: NDArray[np.float32],
    diameters_at_breast_height: NDArray[np.float32],
    heights: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Crown area of tree, Equation (8) of Li ea."""
    t_model_crown_areas = (
        (np.pi * pft_ca_ratio_values / (4 * pft_a_hd_values))
        * diameters_at_breast_height
        * heights
    )

    return t_model_crown_areas


def calculate_crown_fractions(
    heights: NDArray[np.float32],
    pft_a_hd_values: NDArray[np.float32],
    diameters_at_breast_height: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Crown fraction, Equation (11) of Li ea."""
    crown_fractions = heights / (pft_a_hd_values * diameters_at_breast_height)

    return crown_fractions


def calculate_stem_masses(
    diameters_at_breast_height: NDArray[np.float32],
    heights: NDArray[np.float32],
    pft_rho_s_values: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Mass of stems."""
    stem_masses = (
        (np.pi / 8) * (diameters_at_breast_height**2) * heights * pft_rho_s_values
    )

    return stem_masses


def calculate_foliage_masses(
    crown_areas: NDArray[np.float32],
    pft_lai_values: NDArray[np.float32],
    pft_sla_values: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Mass of foliage."""
    foliage_masses = crown_areas * pft_lai_values * (1 / pft_sla_values)

    return foliage_masses


def calculate_swd_masses(
    crown_areas: NDArray[np.float32],
    pft_rho_s_values: NDArray[np.float32],
    heights: NDArray[np.float32],
    crown_fractions: NDArray[np.float32],
    pft_ca_ratio_values: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Mass of ???"""
    swd_masses = (
        crown_areas
        * pft_rho_s_values
        * heights
        * (1 - crown_fractions / 2)
        / pft_ca_ratio_values
    )

    return swd_masses


def calculate_q_m_values(
    m: NDArray[np.float32], n: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Placeholder."""
    return (
        m
        * n
        * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
        * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_z_max_values(
    m: NDArray[np.float32], n: NDArray[np.float32], height: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Calculate z_m, the height of maximum crown radius."""
    # Height of maximum crown radius
    z_max = height * ((n - 1) / (m * n - 1)) ** (1 / n)
    return z_max


def calculate_r_0_values(
    q_m: NDArray[np.float32], crown_area: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Calculate stem canopy factors from Jaideep's extension to the T Model."""
    # Scaling factor to give expected A_c (crown area) at
    # z_m (height of maximum crown radius)
    r_0 = 1 / q_m * np.sqrt(crown_area / np.pi)

    return r_0


def calculate_relative_canopy_radii(
    z: float,
    height: NDArray[np.float32],
    m: NDArray[np.float32],
    n: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate q(z) at a given height, z."""

    z_over_height = z / height

    return m * n * z_over_height ** (n - 1) * (1 - z_over_height**n) ** (m - 1)
