"""Canopy Factors from Jaideep's extension to the T Model.

Utilities to calculate and store the canopy factors from Jaideep's extension to
the T Model.
"""

import numpy as np
from numpy.typing import NDArray


def calculate_q_m(m: NDArray[np.float32],
                  n: NDArray[np.float32]) -> NDArray[np.float32]:
    """placeholder."""
    return (
            m
            * n
            * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
            * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_z_m(m: NDArray[np.float32],
                  n: NDArray[np.float32],
                  height: NDArray[np.float32]) -> NDArray[np.float32]:
    # Height of maximum crown radius
    z_m = height * ((n - 1) / (m * n - 1)) ** (1 / n)
    return z_m


def calculate_r_0(q_m: NDArray[np.float32],
                  crown_area: NDArray[np.float32]
                  ) -> NDArray[np.float32]:
    """Calculate stem canopy factors from Jaideep's extension to the T Model."""
    # Scaling factor to give expected Ac (crown area) at
    # z_m (height of maximum crown radius)
    r_0 = 1 / q_m * np.sqrt(crown_area / np.pi)

    return r_0


def calculate_relative_canopy_radius(z: float, H: float, m: float, n: float) -> float:
    """Calculate q(z) at a given height, z."""

    z_over_H = z / H

    return m * n * z_over_H ** (n - 1) * (1 - z_over_H ** n) ** (m - 1)


calculate_relative_canopy_radius_profile = np.vectorize(
    calculate_relative_canopy_radius
)


def calculate_projected_canopy_area_for_individual(
        z: float,
        pft: PlantFunctionalType,
        t_model_geometry: TModelGeometry,
        canopy_factors: CanopyFactors,
) -> float:
    """Calculate projected crown area above a given height.

    This function takes PFT specific parameters (shape parameters) and stem specific
    sizes and estimates the projected crown area above a given height $z$. Note,
    this calculation gives the canopy area for a single individual within the cohort,
    not for the cohort as a whole.
    :param canopy_factors:
    :param pft:
    :param z_m: stem canopy factor from Jaideep’s extension of the T Model.
    :param q_m: stem canopy factor from Jaideep’s extension of the T Model.
    :param z: height on the z axis.
    :param t_model_geometry: calculated geometry of cohort using T model.
    """

    # Calculate q(z)
    q_z = calculate_relative_canopy_radius(z, t_model_geometry.height, pft.m, pft.n)

    # Calculate A_p
    if z <= canopy_factors.z_m:
        A_p = t_model_geometry.crown_area
    elif z > t_model_geometry.height:
        A_p = 0
    else:
        A_p = t_model_geometry.crown_area * (q_z / canopy_factors.q_m) ** 2

    return A_p


def calculate_crown_radius_profile(
        pft: PlantFunctionalType,
        t_model_geometry: TModelGeometry,
        z_resolution: float = 0.05,
) -> NDArray:
    """Calculate the crown radius profile for a given cohort."""

    z = create_z_axis(0, t_model_geometry.height, z_resolution)

    canopy_factors = calculate_stem_canopy_factors(pft, t_model_geometry)

    # calculate r(z) = r0 * q(z) for a cohort
    r_z = canopy_factors.r_0 * calculate_relative_canopy_radius_profile(
        z, t_model_geometry.height, pft.m, pft.n
    )

    # When z > H, r_z < 0, so set radius to 0 where rz < 0
    r_z[np.where(r_z < 0)] = 0

    return r_z


def create_z_axis(
        z_min: float, z_max: float, resolution: float = 0.05
) -> np.typing.NDArray:
    """Provides a z axis in the form of a numpy array.

    :param z_min: start of the axis
    :param z_max: end of the axis
    :param resolution: resolution of the axis
    :return: a z axis from z_min, to z_max, in increments of the resolution
    """
    max_height_padding = 1
    floating_point_correction = 0.00001

    z = np.arange(z_min, z_max + max_height_padding, resolution)
    z = np.sort(np.concatenate([z, z_max - floating_point_correction]))

    return z
