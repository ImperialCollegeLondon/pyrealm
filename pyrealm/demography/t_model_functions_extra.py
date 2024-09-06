"""Canopy Factors from Jaideep's extension to the T Model.

Utilities to calculate and store the canopy factors from Jaideep's extension to
the T Model.
"""

import numpy as np
from numpy.typing import NDArray


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
