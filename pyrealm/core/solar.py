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


## plan to move individual functions from splash.solar.py here for general use

<<<<<<< Updated upstream

def calc_distance_factor(nu, k_e):
    """Calculate distance factor
=======
def calc_distance_factor(nu: NDArray, k_e: float):
    '''Calculate distance factor
>>>>>>> Stashed changes

    This function calculates distance factor using the method of Berger et al. (1993)
    Args:
        nu: helio centric true anomaly
        k_e: Solar eccentricity

    Returns:
        dr: distance factor
    """
    dr = (1.0 / ((1.0 - k_e**2) / (1.0 + k_e * np.cos(np.deg2rad(nu))))) ** 2

    return dr

<<<<<<< Updated upstream

def calc_declination_angle_delta(lambda_, k_eps, k_pir):
    """Calculate declination angle delta
=======
def calc_declination_angle_delta(lambda_: NDArray, k_eps: float, k_pir: float):
    '''Calculate declination angle delta
>>>>>>> Stashed changes

    This function calculates the solar declination angle delta using the method of Woolf (1968)

    Args:
        lambda_: heliocentric longitude
        k_eps: Solar obliquity
        k_pir: conversion factor from radians to degrees

    Returns:
        delta: solar declination angle delta
    """
    delta = np.arcsin(np.sin(np.deg2rad(lambda_)) * np.sin(np.deg2rad(k_eps))) / k_pir

    return delta

<<<<<<< Updated upstream

def calc_lat_delta_intermediates(delta, lat):
    """Calculates intermediate values for use in solar radiation calcs
=======
def calc_lat_delta_intermediates(delta: NDArray, lat: NDArray):
    '''Calculates intermediate values for use in solar radiation calcs
>>>>>>> Stashed changes

    This function calculates ru and rv which are dimensionless intermediate values calculated from the solar declination angle delta and the observation latitude

    Args:
        delta: solar declination delta
        lat: observation latitude

    Returns:
        Tuple: ru, rv

    """
    ru = np.sin(np.deg2rad(delta)) * np.sin(np.deg2rad(lat))
    rv = np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(lat))

    return ru, rv


def calc_sunset_hour_angle(ru, rv, k_pir):
    """Calculate sunset hour angle

    This function calculates the sunset hour angle using Eq3.22, Stine & Geyer (2001)

    Args:
        ru: dimensionless
    """

    angle = np.arccos(-1.0 * np.clip(ru / rv, -1.0, 1.0)) / k_pir

    return angle


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
