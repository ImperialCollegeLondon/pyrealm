"""To do."""

import numpy as np
from numpy.typing import NDArray


class TwoLeafIrradience:
    """Calculates two-leaf irradience."""

    def __init__(self, beta_angle: NDArray, PPFD: NDArray, LAI: NDArray, PATM: NDArray):

        self.beta_angle: NDArray = beta_angle


def beta_angle(lat: NDArray, d: NDArray, h: NDArray) -> NDArray:
    """Calculates solar beta angle.

    Calculates solar beta angle using Eq A13 of dePury & Farquhar (1997).

    Args:
        lat: array of latitudes (rads)
        d: array of declinations (rads)
        h: array of hour angle (rads)

    Returns:
        beta: array of solar beta angles
    """

    beta = np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(h)

    return beta
