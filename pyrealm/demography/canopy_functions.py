"""A set of functions implementing the canopy shape and vertical leaf distribution model
used in PlantFATE :cite:t:`joshi:2022a`.
"""  # noqa: D205

import numpy as np
from numpy.typing import NDArray
from pandas import Series

# from pyrealm.demography.community import Community


def calculate_canopy_q_m(
    m: float | NDArray[np.float32], n: float | NDArray[np.float32]
) -> NDArray[np.float32]:
    """Calculate the canopy scaling paramater ``q_m``.

    The value of q_m is a constant canopy scaling parameter derived from the ``m`` and
    ``n`` attributes defined for a plant functional type.

    Args:
        m: Canopy shape parameter
        n: Canopy shape parameter
    """
    return (
        m
        * n
        * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
        * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_canopy_z_max_proportion(
    m: float | NDArray[np.float32], n: float | NDArray[np.float32]
) -> NDArray[np.float32]:
    r"""Calculate the z_m proportion.

    The z_m proportion (:math:`p_{zm}`) is the constant proportion of stem height at
    which the maximum crown radius is found for a given plant functional type.

    .. math::

        p_{zm} = \left(\dfrac{n-1}{m n -1}\right)^ {\tfrac{1}{n}}

    Args:
        m: Canopy shape parameter
        n: Canopy shape parameter
    """

    return ((n - 1) / (m * n - 1)) ** (1 / n)


def calculate_canopy_z_max(z_max_prop: Series, stem_height: Series) -> Series:
    r"""Calculate height of maximum crown radius.

    The height of the maximum crown radius (:math:`z_m`) is derived from the canopy
    shape parameters (:math:`m,n`) and the resulting fixed proportion (:math:`p_{zm}`)
    for plant functional types. These shape parameters are defined as part of the
    extension of the T Model presented by :cite:t:`joshi:2022a`.

    The value :math:`z_m` is the height above ground where the largest canopy radius is
    found, given the proportion and the estimated stem height (:math:`H`) of
    individuals.

    .. math::

        z_m = p_{zm} H

    Args:
        z_max_prop: Canopy shape parameter of the PFT
        stem_height: Stem height of individuals
    """
    """Calculate z_m, the height of maximum crown radius."""

    return stem_height * z_max_prop


def calculate_canopy_r0(q_m: Series, crown_area: Series) -> Series:
    r"""Calculate scaling factor for height of maximum crown radius.

    This scaling factor (:math:`r_0`) is derived from the canopy shape parameters
    (:math:`m,n,q_m`) for plant functional types and the estimated crown area
    (:math:`A_c`) of individuals. The shape parameters are defined as part of the
    extension of the T Model presented by :cite:t:`joshi:2022a` and :math:`r_0` is used
    to scale the crown area such that the crown area at the  maximum crown radius fits
    the expectations of the T Model.

    .. math::

        r_0 = 1/q_m  \sqrt{A_c / \pi}

    Args:
        q_m: Canopy shape parameter of the PFT
        crown_area: Crown area of individuals
    """
    # Scaling factor to give expected A_c (crown area) at
    # z_m (height of maximum crown radius)

    return 1 / q_m * np.sqrt(crown_area / np.pi)


def calculate_relative_canopy_radius_at_z(
    z: float,
    stem_height: NDArray[np.float32],
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
        stem_height: Total height of individual stem
        m: Canopy shape parameter of PFT
        n: Canopy shape parameter of PFT
    """

    z_over_height = z / stem_height

    return m * n * z_over_height ** (n - 1) * (1 - z_over_height**n) ** (m - 1)


def calculate_stem_projected_canopy_area_at_z(
    z: float,
    stem_height: NDArray[np.float32],
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
        stem_height: Stem height of each cohort
        m: Canopy shape parameter ``m``` for each cohort
        n: Canopy shape parameter ``n``` for each cohort
        q_m: Canopy shape parameter ``q_m``` for each cohort
        z_m: Canopy shape parameter ``z_m``` for each cohort
    """

    # Calculate q(z)
    q_z = calculate_relative_canopy_radius_at_z(z, stem_height, m, n)

    # Calculate A_p
    # Calculate Ap given z > zm
    A_p = crown_area * (q_z / q_m) ** 2
    # Set Ap = Ac where z <= zm
    A_p = np.where(z <= z_m, crown_area, A_p)
    # Set Ap = 0 where z > H
    A_p = np.where(z > stem_height, 0, A_p)

    return A_p


def solve_community_projected_canopy_area(
    z: float,
    stem_height: NDArray[np.float32],
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
        stem_height: Stem height of each cohort
        m: Canopy shape parameter ``m``` for each cohort
        n: Canopy shape parameter ``n``` for each cohort
        q_m: Canopy shape parameter ``q_m``` for each cohort
        z_m: Canopy shape parameter ``z_m``` for each cohort
        target_area: A target projected crown area.
    """

    # Calculate A(p) for the stems in each cohort
    A_p = calculate_stem_projected_canopy_area_at_z(
        z=z,
        stem_height=stem_height,
        crown_area=crown_area,
        m=m,
        n=n,
        q_m=q_m,
        z_m=z_m,
    )

    return (A_p * n_individuals).sum() - target_area


def calculate_stem_projected_leaf_area_at_z(
    z: float | NDArray[np.float32],
    stem_height: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    z_max: NDArray[np.float32],
    f_g: NDArray[np.float32],
    m: NDArray[np.float32],
    n: NDArray[np.float32],
    q_m: NDArray[np.float32],
    z_m: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate projected leaf area above a given height.

    Calculation applies to an individual within a cohort.This function takes PFT
    specific parameters (shape parameters) and stem specific sizes and estimates
    the projected crown area above a given height $z$. The inputs can either be
    scalars describing a single stem or arrays representing a community of stems.
    If only a single PFT is being modelled then `m`, `n`, `qm` and `fg` can be
    scalars with arrays `H`, `Ac` and `zm` giving the sizes of stems within that
    PFT.

    Args:
        z: Vertical height on the z axis.
        crown_area: Crown area for a stem
        stem_height: Total height of a stem
        z_max: Height of maximum canopy radius for each stem
        f_g: Within crown gap fraction for each stem.
        m: Canopy shape parameter ``m``` for each stem
        n: Canopy shape parameter ``n``` for each stem
        q_m: Canopy shape parameter ``q_m``` for each stem
        z_m: Canopy shape parameter ``z_m``` for each stem
    """

    # Calculate q(z)
    q_z = calculate_relative_canopy_radius_at_z(
        z=z,
        stem_height=stem_height,
        m=m,
        n=n,
    )

    # Calculate Ac terms
    A_c_terms = crown_area * (q_z / q_m) ** 2

    # Set Acp either side of zm
    A_cp = np.where(
        z <= z_max,
        crown_area - A_c_terms * f_g,
        A_c_terms * (1 - f_g),
    )
    # Set Ap = 0 where z > H
    A_cp = np.where(z > stem_height, 0, A_cp)

    return A_cp


# def calculate_total_canopy_A_cp(z: float, f_g: float, community: Community) -> float:
#     """Calculate total leaf area at a given height.

#     :param f_g:
#     :param community:
#     :param z: Height above ground.
#     :return: Total leaf area in the canopy at a given height.
#     """
#     A_cp_for_individuals = calculate_projected_leaf_area_for_individuals(
#         z, f_g, community
#     )

#     A_cp_for_cohorts = A_cp_for_individuals * community.cohort_number_of_individuals

#     return A_cp_for_cohorts.sum()


# def calculate_gpp(cell_ppfd: NDArray, lue: NDArray) -> float:
#     """Estimate the gross primary productivity.

#     Not sure where to place this - need an array of LUE that matches to the

#     """

#     return 100
