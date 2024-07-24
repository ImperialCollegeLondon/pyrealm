"""Populate t model arrays.

Populate the relevant initialised arrays with properties calculated using the
t model.
:return: None
"""

import numpy as np
from numpy.typing import NDArray


def calculate_heights(
    pft_h_max_values: NDArray[np.float32],
    pft_a_hd_values: NDArray[np.float32],
    diameters_at_breast_height: NDArray[np.float32],
) -> NDArray[np.float32]:
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
    # Crown area of tree, Equation (8) of Li ea.
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
    # Crown fraction, Equation (11) of Li ea.
    crown_fractions = heights / (pft_a_hd_values * diameters_at_breast_height)

    return crown_fractions


def calculate_stem_masses(
    diameters_at_breast_height: NDArray[np.float32],
    heights: NDArray[np.float32],
    pft_rho_s_values: NDArray[np.float32],
) -> NDArray[np.float32]:
    stem_masses = (
        (np.pi / 8) * (diameters_at_breast_height**2) * heights * pft_rho_s_values
    )

    return stem_masses


def calculate_foliage_masses(
    crown_areas: NDArray[np.float32],
    pft_lai_values: NDArray[np.float32],
    pft_sla_values: NDArray[np.float32],
) -> NDArray[np.float32]:
    foliage_masses = crown_areas * pft_lai_values * (1 / pft_sla_values)

    return foliage_masses


def calculate_swd_masses(
    crown_areas, pft_rho_s_values, heights, crown_fractions, pft_ca_ratio_values
):
    swd_masses = (
        crown_areas
        * pft_rho_s_values
        * heights
        * (1 - crown_fractions / 2)
        / pft_ca_ratio_values
    )

    return swd_masses
