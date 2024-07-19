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
    # Scaling factor to give expected A_c (crown area) at
    # z_m (height of maximum crown radius)
    r_0 = 1 / q_m * np.sqrt(crown_area / np.pi)

    return r_0


def calculate_projected_canopy_area_for_individuals(
        z: float,
        height: NDArray[np.float32],
        crown_area: NDArray[np.float32],
        m: NDArray[np.float32],
        n: NDArray[np.float32],
        q_m: NDArray[np.float32],
        z_m: NDArray[np.float32]
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
    :param z_m: stem canopy factor from Jaideep’s extension of the T Model.
    :param q_m: stem canopy factor from Jaideep’s extension of the T Model.
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


def calculate_relative_canopy_radii(z: float,
                                    height: NDArray[np.float32],
                                    m: NDArray[np.float32],
                                    n: NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculate q(z) at a given height, z."""

    z_over_height = z / height

    return m * n * z_over_height ** (n - 1) * (1 - z_over_height ** n) ** (m - 1)

