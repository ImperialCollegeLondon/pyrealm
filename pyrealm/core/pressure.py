"""The :mod:`~pyrealm.core.pressure` submodule contains core functions for calculating
atmospheric pressure.
"""  # noqa D210, D415

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst


def calc_patm(
    elv: NDArray[np.float64], core_const: CoreConst = CoreConst()
) -> NDArray[np.float64]:
    r"""Calculate atmospheric pressure from elevation.

    Calculates atmospheric pressure as a function of elevation with reference to the
    standard atmosphere.  The elevation-dependence of atmospheric pressure is computed
    by assuming a linear decrease in temperature with elevation and a mean adiabatic
    lapse rate (Eqn 3, :cite:alp:`BerberanSantos:2009bk`):

    .. math::

        p(z) = p_0 ( 1 - L z / K_0) ^{ G M / (R L) },

    Args:
        elv: Elevation above sea-level (:math:`z`, metres above sea level.)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.

    Returns:
        A numeric value for :math:`p` in Pascals.

    Examples:
        >>> # Standard atmospheric pressure, in Pa, corrected for 1000 m.a.s.l.
        >>> round(calc_patm(1000), 2)
        90241.54
    """

    # Convert elevation to pressure, Pa.
    return core_const.k_Po * (1.0 - core_const.k_L * elv / core_const.k_To) ** (
        core_const.k_G * core_const.k_Ma / (core_const.k_R * core_const.k_L)
    )
