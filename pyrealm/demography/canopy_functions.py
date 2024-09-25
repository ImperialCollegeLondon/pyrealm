"""A set of functions implementing the canopy shape and vertical leaf distribution model
used in PlantFATE :cite:t:`joshi:2022a`.
"""  # noqa: D205

import numpy as np
from numpy.typing import NDArray

from pyrealm.core.utilities import check_input_shapes


def calculate_canopy_q_m(
    m: float | NDArray[np.float32], n: float | NDArray[np.float32]
) -> float | NDArray[np.float32]:
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
) -> float | NDArray[np.float32]:
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


def calculate_canopy_z_max(
    z_max_prop: NDArray[np.float32], stem_height: NDArray[np.float32]
) -> NDArray[np.float32]:
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


def calculate_canopy_r0(
    q_m: NDArray[np.float32], crown_area: NDArray[np.float32]
) -> NDArray[np.float32]:
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


def _validate_z_args(z: NDArray[np.float32], *args: NDArray[np.float32]) -> None:
    """Shared validation routine for vertical height arguments.

    The canopy functions that take a height argument ``z`` can take a range of input
    shapes for ``z``, alongside a set of other row arrays representing cohort
    properties. This function is used to check that the cohort properties are of equal
    length and that the ``z`` value has one of the accepted shapes described in
    :meth:`~pyrealm.demography.canopy_functions.calculate_relative_canopy_radius_at_z`.

    Args:
        z: The inputs to the ``z`` argument of a function.
        args: Other arrays representing cohort properties.
    """

    if z.size == 1 or z.ndim == 1:
        # All same height or stem specific heights - check z is either a scalar or also
        # a row vector of the right length.
        check_input_shapes(z, *args)
        return
    elif z.ndim == 2 and z.shape[1] == 1:
        # Z is a column vector, just check stem properties.
        check_input_shapes(*args)
        return

    raise ValueError("Invalid shape for the z value.")


def _validate_q_z(
    z: NDArray[np.float32],
    q_z: NDArray[np.float32],
    stem_property: NDArray[np.float32],
) -> None:
    """Shared validation routine for relative radius arguments.

    The functions
    :meth:`~pyrealm.demography.canopy_functions.calculate_stem_projected_canopy_area_at_z`
    and
    :meth:`~pyrealm.demography.canopy_functions.calculate_stem_projected_leaf_area_at_z`
    both require the arguments ``z`` and ``q_z``, where ``z`` is an array of vertical
    heights and ``q_z`` is the relative canopy radius at those heights for a set of
    stems. This function checks that the inputs are congruent with each other, and with
    the shape of a stem property argument,  given the set of expected forms of ``z``
    described in
    :meth:`~pyrealm.demography.canopy_functions.calculate_relative_canopy_radius_at_z`.


    Args:
        z: The input to the ``z`` argument.
        q_z: The input to the ``q_z`` argument.
        stem_property: An argument input representing a stem property.
    """

    if z.size == 1 or z.ndim == 1:
        # All same height or stem specific heights - q_z must then also be row vector of
        # the same length as the stem propertys and z must be scalar or a row_vector.
        check_input_shapes(z, q_z, stem_property)
        return
    elif z.ndim == 2 and z.shape[1] == 1:
        # z is a column array, so check q_z is a matrix
        if q_z.shape != (z.size, stem_property.size):
            raise ValueError("Invalid shape for q_z.")
        return

    raise ValueError("Invalid shape for the z value.")


def calculate_relative_canopy_radius_at_z(
    z: NDArray[np.float32],
    stem_height: NDArray[np.float32],
    m: NDArray[np.float32],
    n: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
    r"""Calculate relative canopy radius at a given height.

    The canopy shape parameters ``m`` and ``n`` define the vertical distribution of
    canopy along the stem. For a stem of a given total height, this function calculates
    the relative canopy radius at a given height :math:`z`:

    .. math::

        q(z) = m n \left(\dfrac{z}{H}\right) ^ {n -1}
        \left( 1 - \left(\dfrac{z}{H}\right) ^ n \right)^{m-1}

    This function calculates :math:`q(z)` across a set of stems: the ``stem_height``,
    ``m`` and ``n`` arguments should be one-dimensional arrays ('row vectors') of equal
    length :math:`I`.  The value for ``z`` is then an array of heights, with one of the
    following shapes:

    1. A scalar array: :math:`q(z)` is found for all stems at the same height and the
       return value is a 1D array of length :math:`I`.
    2. A row vector of length :math:`I`: :math:`q(z)` is found for all stems at
       stem-specific heights and the return value is again a 1D array of length
       :math:`I`.
    3. A column vector of length :math:`J`, that is a 2 dimensional array of shape
       (:math:`J`, 1). This allows :math:`q(z)` to be calculated efficiently for a set
       of heights for all stems and return a 2D array of shape (:math:`J`, :math:`I`).

    Args:
        z: Height at which to calculate relative radius
        stem_height: Total height of individual stem
        m: Canopy shape parameter of PFT
        n: Canopy shape parameter of PFT
        validate: Boolean flag to suppress argument validation.
    """

    if validate:
        _validate_z_args(z, stem_height, m, n)

    z_over_height = z / stem_height

    return m * n * z_over_height ** (n - 1) * (1 - z_over_height**n) ** (m - 1)


def calculate_stem_projected_crown_area_at_z(
    z: NDArray[np.float32],
    q_z: NDArray[np.float32],
    stem_height: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    q_m: NDArray[np.float32],
    z_max: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
    """Calculate stem projected crown area above a given height.

    This function calculates the projected crown area of a set of stems with given
    properties at a set of vertical heights. The stem properties are given in the
    arguments ``stem_height``,``crown_area``,``q_m`` and ``z_max``, which must be
    one-dimensional arrays ('row vectors') of equal length. The array of vertical
    heights ``z`` accepts a range of input shapes (see
    :meth:`~pyrealm.demography.canopy_functions.calculate_relative_canopy_radius_at_z`
    ) and this function then also requires the expected relative stem radius (``q_z``)
    calculated from those heights.

    Args:
        z: Vertical height at which to estimate crown area
        q_z: Relative crown radius at those heights
        crown_area: Crown area of each stem
        stem_height: Stem height of each stem
        q_m: Canopy shape parameter ``q_m``` for each stem
        z_max: Height of maximum crown radous for each stem
        validate: Boolean flag to suppress argument validation.
    """

    if validate:
        _validate_z_args(z, stem_height, crown_area, q_m, z_max)
        _validate_q_z(z, q_z, crown_area)

    # Calculate A_p
    # Calculate Ap given z > zm
    A_p = crown_area * (q_z / q_m) ** 2
    # Set Ap = Ac where z <= zm
    A_p = np.where(z <= z_max, crown_area, A_p)
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
    z_max: NDArray[np.float32],
    n_individuals: NDArray[np.float32],
    target_area: float = 0,
    validate: bool = True,
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
        z_max: Canopy shape parameter ``z_m``` for each cohort
        target_area: A target projected crown area.
        validate: Boolean flag to suppress argument validation.
    """
    # Convert z to array for validation and typing
    z_arr = np.array(z)

    if validate:
        _validate_z_args(
            z_arr, n_individuals, crown_area, stem_height, m, n, q_m, z_max
        )

    q_z = calculate_relative_canopy_radius_at_z(
        z=z_arr, stem_height=stem_height, m=m, n=n, validate=False
    )
    # Calculate A(p) for the stems in each cohort
    A_p = calculate_stem_projected_crown_area_at_z(
        z=z_arr,
        q_z=q_z,
        stem_height=stem_height,
        crown_area=crown_area,
        q_m=q_m,
        z_max=z_max,
        validate=False,
    )

    return (A_p * n_individuals).sum() - target_area


def calculate_stem_projected_leaf_area_at_z(
    z: NDArray[np.float32],
    q_z: NDArray[np.float32],
    stem_height: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    f_g: NDArray[np.float32],
    q_m: NDArray[np.float32],
    z_max: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
    """Calculate projected leaf area above a given height.

    This function calculates the projected leaf area of a set of stems with given
    properties at a set of vertical heights. This differs from crown area in allowing
    for crown openness within the crown of an individual stem that results in the
    displacement of leaf area further down into the canopy. The degree of openness is
    controlled by the crown gap fraction property of each stem.

    The stem properties are given in the arguments
    ``stem_height``,``crown_area``,``f_g``,``q_m`` and ``z_max``, which must be
    one-dimensional arrays ('row vectors') of equal length. The array of vertical
    heights ``z`` accepts a range of input shapes (see
    :meth:`~pyrealm.demography.canopy_functions.calculate_relative_canopy_radius_at_z`
    ) and this function then also requires the expected relative stem radius (``q_z``)
    calculated from those heights.

    Args:
        z: Vertical heights on the z axis.
        q_z: Relative crown radius at heights in z.
        crown_area: Crown area for a stem
        stem_height: Total height of a stem
        f_g: Within crown gap fraction for each stem.
        q_m: Canopy shape parameter ``q_m``` for each stem
        z_max: Height of maximum canopy radius for each stem
        validate: Boolean flag to suppress argument validation.
    """

    # NOTE: Although the internals of this function overlap a lot with
    #       calculate_stem_projected_crown_area_at_z, we want that function to be as
    #       lean as possible, as it used within solve_community_projected_canopy_area.

    if validate:
        _validate_z_args(z, crown_area, stem_height, f_g, q_m, z_max)
        _validate_q_z(z, q_z, crown_area)

    # Calculate Ac terms
    A_c_terms = crown_area * (q_z / q_m) ** 2

    # Set Acp either side of z_max
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
