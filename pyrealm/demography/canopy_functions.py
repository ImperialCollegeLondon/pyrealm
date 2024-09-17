"""Class containing functions for calculating properties of a canopy."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from pyrealm.demography.community import Community
from pyrealm.demography.t_model_functions import calculate_relative_canopy_radii


def calculate_total_community_crown_area(community: Community) -> float:
    """Calculate the total crown area of a community."""
    # Calculate the number of layers
    cohort_crown_areas = (
        community.cohort_number_of_individuals * community.t_model_crown_areas
    )
    total_community_crown_area = cohort_crown_areas.sum()
    return total_community_crown_area


def calculate_number_of_canopy_layers(
    cell_area: float, total_community_crown_area: float, fG: float
) -> int:
    """Calculate the number of canopy layers in a given community."""
    number_of_layers = int(np.ceil(total_community_crown_area / (cell_area * (1 - fG))))
    return number_of_layers


def calculate_community_projected_area_at_z(community: Community, z: float) -> float:
    """Calculate the total area of community stems."""
    projected_canopy_area_for_individuals = (
        calculate_projected_canopy_area_for_individuals(
            z,
            community.t_model_heights,
            community.t_model_crown_areas,
            community.pft_m_values,
            community.pft_n_values,
            community.canopy_factor_q_m_values,
            community.canopy_factor_z_m_values,
        )
    )

    cohort_areas_at_z = (
        community.cohort_number_of_individuals * projected_canopy_area_for_individuals
    )

    return sum(cohort_areas_at_z)


def calculate_projected_canopy_area_for_individuals(
    z: float,
    height: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    m: NDArray[np.float32],
    n: NDArray[np.float32],
    q_m: NDArray[np.float32],
    z_m: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate projected crown area above a given height.

    This function takes PFT specific parameters (shape parameters) and stem specific
    sizes and estimates the projected crown area above a given height $z$. Note,
    this calculation gives the canopy area for a single individual within the cohort,
    not for the cohort as a whole.
    :param m:
    :param n:
    :param crown_area:
    :param height:
    :param z_m: stem canopy factor from Jaideep extension of the T Model.
    :param q_m: stem canopy factor from Jaideep extension of the T Model.
    :param z: height on the z axis.
    """

    # Calculate q(z)
    q_z = calculate_relative_canopy_radii(z, height, m, n)

    # Calculate A_p
    # Calculate Ap given z > zm
    A_p = crown_area * (q_z / q_m) ** 2
    # Set Ap = Ac where z <= zm
    A_p = np.where(z <= z_m, crown_area, A_p)
    # Set Ap = 0 where z > H
    A_p = np.where(z > height, 0, A_p)

    return A_p


def calculate_canopy_layer_heights(
    number_of_canopy_layers: int,
    max_individual_height: float,
    community: Community,
    fG: float,
) -> NDArray:
    """Calculate the heights of the layers of the canopy for a community."""

    # Data store for z*
    z_star = np.zeros(number_of_canopy_layers)

    # Loop over the layers TODO - edge case of completely filled final layer
    for n in np.arange(number_of_canopy_layers - 1):
        z_star[n] = root_scalar(
            solve_canopy_closure_height,
            args=(community, n + 1, community.cell_area, fG),
            bracket=(0, max_individual_height),
        ).root

    return z_star


def solve_canopy_closure_height(
    z: float,
    community: Community,
    layer_index: int,
    A: float,
    fG: float,
) -> float:
    """Solver function for canopy closure height.

    This function returns the difference between the total community projected area
    at a height $z$ and the total available canopy space for canopy layer $l$, given
    the community gap fraction for a given height. It is used with a root solver to
    find canopy layer closure heights $z^*_l* for a community.
    :param community:
    :param fG: community gap fraction
    :param A: community area
    :param layer_index: layer index
    :param z: height
    """

    community_projected_area_at_z = calculate_community_projected_area_at_z(
        community, z
    )

    # Return the difference between the projected area and the available space
    return community_projected_area_at_z - (A * layer_index) * (1 - fG)


def calculate_projected_leaf_area_for_individuals(
    z: float, f_g: float, community: Community
) -> NDArray[np.float32]:
    """Calculate projected crown area above a given height.

    Calculation applies to an individual within a cohort.This function takes PFT
    specific parameters (shape parameters) and stem specific sizes and estimates
    the projected crown area above a given height $z$. The inputs can either be
    scalars describing a single stem or arrays representing a community of stems.
    If only a single PFT is being modelled then `m`, `n`, `qm` and `fg` can be
    scalars with arrays `H`, `Ac` and `zm` giving the sizes of stems within that
    PFT.

    :param community:
    :param f_g: Crown gap fraction.
    :param z: Height from ground.
    """

    # Calculate q(z)
    q_z = calculate_relative_canopy_radii(
        z, community.t_model_heights, community.pft_m_values, community.pft_n_values
    )

    # Calculate Ac terms
    A_c_terms = (
        community.t_model_crown_areas * (q_z / community.canopy_factor_q_m_values) ** 2
    )

    # Set Acp either side of zm
    A_cp = np.where(
        z <= community.canopy_factor_z_m_values,
        community.t_model_crown_areas - A_c_terms * f_g,
        A_c_terms * (1 - f_g),
    )
    # Set Ap = 0 where z > H
    A_cp = np.where(z > community.t_model_heights, 0, A_cp)
    return A_cp


def calculate_total_canopy_A_cp(z: float, f_g: float, community: Community) -> float:
    """Calculate total leaf area at a given height.

    :param f_g:
    :param community:
    :param z: Height above ground.
    :return: Total leaf area in the canopy at a given height.
    """
    A_cp_for_individuals = calculate_projected_leaf_area_for_individuals(
        z, f_g, community
    )

    A_cp_for_cohorts = A_cp_for_individuals * community.cohort_number_of_individuals

    return A_cp_for_cohorts.sum()


def calculate_gpp(cell_ppfd: NDArray, lue: NDArray) -> float:
    """Estimate the gross primary productivity.

    Not sure where to place this - need an array of LUE that matches to the

    """

    return 100
