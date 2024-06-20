"""The ``solar`` submodule provides functions and classes to calculate daily solar
radiation fluxes and other radiative values.
"""  # noqa: D205

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst

# from pyrealm.splash.const import (
#     kA,
#     kalb_sw,
#     kalb_vis,
#     kb,
#     kc,
#     kd,
#     ke,
#     keps,
#     kfFEC,
#     kGsc,
#     komega,
#     pir,
# )


def calc_heliocentric_longitudes(
    julian_day: NDArray, n_days: NDArray, core_const: CoreConst = CoreConst()
) -> tuple[NDArray, NDArray]:
    """Calculate heliocentric longitude and anomaly.

    This function calculates the heliocentric true anomaly (``nu``, degrees) and true
    longitude (``lambda_``, degrees), given the Julian day in the year and the number of
    days in the year, following :cite:t:`berger:1978a`.

    Args:
        julian_day: day of year
        n_days: number of days in year
        core_const: An instance of CoreConst.

    Returns:
        A tuple of arrays containing ``nu`` and ``lambda_``.
    """

    # Variable substitutes:
    xee = core_const.k_e**2
    xec = core_const.k_e**3
    xse = np.sqrt(1.0 - xee)

    # Mean longitude for vernal equinox:
    xlam = (
        (
            (
                (core_const.k_e / 2.0 + xec / 8.0)
                * (1.0 + xse)
                * np.sin(np.deg2rad(core_const.k_omega))
            )
            - (xee / 4.0 * (0.5 + xse) * np.sin(np.deg2rad(2.0 * core_const.k_omega)))
            + (
                xec
                / 8.0
                * (1.0 / 3.0 + xse)
                * np.sin(np.deg2rad(3.0 * core_const.k_omega))
            )
        )
        * 2.0
        / core_const.k_pir
    )

    # Mean longitude for day of year:
    dlamm = xlam + (julian_day - 80.0) * (360.0 / n_days)

    # Mean anomaly:
    ranm = (dlamm - core_const.k_omega) * core_const.k_pir

    # True anomaly:
    ranv = (
        ranm
        + ((2.0 * core_const.k_e - xec / 4.0) * np.sin(ranm))
        + (5.0 / 4.0 * xee * np.sin(2.0 * ranm))
        + (13.0 / 12.0 * xec * np.sin(3.0 * ranm))
    )

    # True longitude in degrees constrained to 0 - 360
    lambda_ = ((ranv / core_const.k_pir) + core_const.k_omega) % 360

    # True anomaly in degrees constrained to 0 - 360
    nu = (lambda_ - core_const.k_omega) % 360

    return (nu, lambda_)
