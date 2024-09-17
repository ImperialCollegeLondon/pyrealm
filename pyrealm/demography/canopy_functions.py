"""Class containing functions for calculating properties of a canopy."""

import numpy as np
from numpy.typing import NDArray

from pyrealm.demography.community import Community


def calculate_relative_canopy_radius_at_z(
    z: float,
    height: NDArray[np.float32],
    m: NDArray[np.float32],
    n: NDArray[np.float32],
) -> NDArray[np.float32]:
    r"""Calculate relative canopy radius at a given height.

    The canopy shape parameters ``m`` and ``n`` define the vertical distribution of
    canopy along the stem. For a stem of a given total height, this function calculates
    the relative canopy radius at a given height :math:`z`:

    .. math::

        q(z) = m n \left(\dfrac{z}{H}\right) ^ {n -1}
        \left( 1 - \left(\dfrac{z}{H}\right) ^ n \right)^{m-1}

    Args:
        z: Height at which to calculate relative radius
        height: Total height of individual stem
        m: Canopy shape parameter of PFT
        n: Canopy shape parameter of PFT
    """

    z_over_height = z / height

    return m * n * z_over_height ** (n - 1) * (1 - z_over_height**n) ** (m - 1)


def calculate_stem_projected_canopy_area_at_z(
    z: float,
    height: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    m: NDArray[np.float32],
    n: NDArray[np.float32],
    q_m: NDArray[np.float32],
    z_m: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate stem projected crown area above a given height.

    This function takes data on stem heights and crown areas, and then uses the canopy
    shape parameters associated with each stem to calculate the projected crown area
    above a given height $z$.

    Args:
        z: Vertical height on the z axis.
        crown_area: Crown area of each cohort
        height: Stem height of each cohort
        m: Canopy shape parameter ``m``` for each cohort
        n: Canopy shape parameter ``n``` for each cohort
        q_m: Canopy shape parameter ``q_m``` for each cohort
        z_m: Canopy shape parameter ``z_m``` for each cohort
    """

    # Calculate q(z)
    q_z = calculate_relative_canopy_radius_at_z(z, height, m, n)

    # Calculate A_p
    # Calculate Ap given z > zm
    A_p = crown_area * (q_z / q_m) ** 2
    # Set Ap = Ac where z <= zm
    A_p = np.where(z <= z_m, crown_area, A_p)
    # Set Ap = 0 where z > H
    A_p = np.where(z > height, 0, A_p)

    return A_p


def solve_community_projected_canopy_area(
    z: float,
    height: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    m: NDArray[np.float32],
    n: NDArray[np.float32],
    q_m: NDArray[np.float32],
    z_m: NDArray[np.float32],
    n_individuals: NDArray[np.float32],
    target_area: float = 0,
) -> NDArray[np.float32]:
    """Solver function for community wide projected crown area.

    This function takes the number of individuals in each cohort along with the stem
    height and crown area and a given vertical height (:math:`z`). It then uses the
    canopy shape parameters associated with each cohort to calculate the community wide
    projected crown area above that height (:math:`A_p(z)`). This is simply the sum of
    the products of the individual stem projected area at :math:`z` and the number of
    individuals in each cohort.

    The return value is the difference between the calculated :math:`A_p(z)` and a
    user-specified target area, This allows the function to be used with a root solver
    to find :math:`z` values that result in a given :math:`A_p(z)`. The default target
    area is zero, so the default return value will be the actual total :math:`A_p(z)`
    for the community.

    A typical use case for the target area would be to specify the area at which a given
    canopy layer closes under the perfect plasticity approximation in order to find the
    closure height.

    Args:
        z: Vertical height on the z axis.
        n_individuals: Number of individuals in each cohort
        crown_area: Crown area of each cohort
        height: Stem height of each cohort
        m: Canopy shape parameter ``m``` for each cohort
        n: Canopy shape parameter ``n``` for each cohort
        q_m: Canopy shape parameter ``q_m``` for each cohort
        z_m: Canopy shape parameter ``z_m``` for each cohort
        target_area: A target projected crown area.
    """

    # Calculate A(p) for the stems in each cohort
    A_p = calculate_stem_projected_canopy_area_at_z(
        z=z,
        height=height,
        crown_area=crown_area,
        m=m,
        n=n,
        q_m=q_m,
        z_m=z_m,
    )

    return (A_p * n_individuals).sum() - target_area


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
