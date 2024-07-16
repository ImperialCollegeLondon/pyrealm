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


# plan to move individual functions from splash.solar.py here for general use


def calc_distance_factor(nu: NDArray, k_e: float) -> NDArray:
    """Calculate distance factor.

    This function calculates distance factor using the method of Berger et al. (1993)

    Args:
        nu: heliocentric true anomaly (degrees)
        k_e: Solar eccentricity

    Returns:
        dr: distance factor
    """

    dr = (1.0 / ((1.0 - k_e**2) / (1.0 + k_e * np.cos(np.deg2rad(nu))))) ** 2

    return dr


def calc_declination_angle_delta(
    lambda_: NDArray, k_eps: float, k_pir: float
) -> NDArray:
    """Calculate declination angle delta.

    This function calculates the solar declination angle delta using
    the method of Woolf (1968)

    Args:
        lambda_: heliocentric longitude
        k_eps: Solar obliquity
        k_pir: conversion factor from radians to degrees

    Returns:
        delta: solar declination angle delta
    """
    delta = np.arcsin(np.sin(np.deg2rad(lambda_)) * np.sin(np.deg2rad(k_eps))) / k_pir

    return delta


def calc_lat_delta_intermediates(
    delta: NDArray, lat: NDArray
) -> tuple[NDArray, NDArray]:
    """Calculates intermediate values for use in solar radiation calcs.

    This function calculates ru and rv which are dimensionless intermediate
    values calculated from the solar declination angle delta and the
    observation latitude.

    Args:
        delta: solar declination delta
        lat: observation latitude

    Returns:
        Tuple: ru, rv

    """
    ru = np.sin(np.deg2rad(delta)) * np.sin(np.deg2rad(lat))
    rv = np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(lat))

    return ru, rv


def calc_sunset_hour_angle(ru: NDArray, rv: NDArray, k_pir: float) -> NDArray:
    """Calculate sunset hour angle.

    This function calculates the sunset hour angle using Eq3.22, Stine & Geyer (2001)

    Args:
        ru: dimensionless parameter
        rv: dimensionless parameter
        k_pir: constant rad to degrees conversion, degrees/rad
    """

    angle = np.arccos(-1.0 * np.clip(ru / rv, -1.0, 1.0)) / k_pir

    return angle


def calc_daily_solar_radiation(
    rad_const: float, dr: NDArray, ru: NDArray, rv: NDArray, k_pir: float, hs: NDArray
) -> NDArray:
    """Calculate daily extraterrestrial solar radiation (J/m^2).

    This function calculates the daily extraterrestrial solar radition (J/m^2)
    using Eq. 1.10.3, Duffy & Beckman (1993)

    Args:
        rad_const: planetary radiation constant, W/m^2
        dr: dimensionless distance factor
        ru: dimensionless variable substitute
        rv: dimensionless variable substitute
        k_pir: radians to degrees conversion, degrees/rad
        hs: local hour angle, degrees

    Returns:
        ra_d: daily solar radiation, J/m^2
    """

    secs_d = 86400  # seconds in one earth day

    ra_d = (
        (secs_d / np.pi)
        * rad_const
        * dr
        * (ru * k_pir * hs + rv * np.sin(np.deg2rad(hs)))
    )

    return ra_d


def calc_transmissivity(k_c: float, k_d: float, sf: NDArray, elv: NDArray) -> NDArray:
    """Calculate atmospheric transmissivity, tau.

    This function calculates atmospheric transmissivity using the method of
    Eq.11, Linacre (1968) and Eq 2, Allen (1996)

    Args:
        k_c: dimensionless cloudy transmissivity
        k_d: dimensionless angular coefficient of transmissivity
        sf: Daily sunshine fraction of observations, unitless
        elv: Elevation of observations, metres

    Returns:
        tau: bulk transmissivity, unitless
    """
    tau = (k_c + k_d * sf) * (1.0 + (2.67e-5) * elv)
    return tau


def calc_ppfd(k_fFEC: float, k_alb_vis: float, tau: NDArray, ra_d: NDArray) -> NDArray:
    """Calculate photosynthetic photon flux density, PPFD,(mol/m^2).

    This function calculates the PPFD in mol/m^2 from secondary calculated
    variables and constants.

    Args:
        k_fFEC: from flux to energy conversion, umol/J
        k_alb_vis: visible light albedo
        tau: bulk transmissivity, unitless
        ra_d: daily solar radiation, J/m^2

    Returns:
        ppfd: photosynthetic photon flux density, mol/m^2
    """

    ppfd = (1.0e-6) * k_fFEC * (1.0 - k_alb_vis) * tau * ra_d

    return ppfd


def calc_rnl(k_b: float, sf: NDArray, k_A: float, tc: NDArray) -> NDArray:
    """Calculates net longwave radiation, rnl, W/m^2.

    This function calculates net longwave radiation in W/m^2.

    Args:
        k_b: calculation constant for Rnl
        sf: sunshine fraction of observations, unitless
        k_A: calculation constant for Rnl
        tc: temperature of observations, °C

    Returns:
        rnl: net longwave radiation, W/m^2
    """

    rnl = (k_b + (1.0 - k_b) * sf) * (k_A - tc)

    return rnl


def calc_rw(k_alb_sw: float, tau: NDArray, k_Gsc: float, dr: NDArray) -> NDArray:
    """Calculates variable substitute rw, W/m^2.

    Args:
        k_alb_sw: shortwave albedo
        tau: bulk transmissivity, unitless
        k_Gsc: solar constant, W/m^2
        dr: distance ration, unitless

    Returns:
        rw: intermediate variable, W/m^2
    """

    rw = (1.0 - k_alb_sw) * tau * k_Gsc * dr

    return rw


def calc_net_rad_crossover_hour_angle(
    rnl: NDArray, rw: NDArray, ru: NDArray, rv: NDArray, k_pir: float
) -> NDArray:
    """Calculates the net radiation crossover hour angle, degrees.

    This function calculates the net radiation crossover hour angle in degrees.

    Args:
        rnl: net longwave radiation, W/m^2
        rw: dimensionless variable substitute
        ru: dimensionless variable substitute
        rv: dimensionless variable substitute
        k_pir: conversion factor from radians to degrees

    Returns:
        hn: crossover hour angle, degrees
    """

    hn = np.arccos(np.clip((rnl - rw * ru) / (rw * rv), -1.0, 1.0)) / k_pir

    return hn


def calc_daytime_net_radiation(
    hn: NDArray, k_pir: float, rw: NDArray, ru: NDArray, rv: NDArray, rnl: NDArray
) -> NDArray:
    """Calculates daily net radiation, J/m^2.

    Args:
        hn: crossover hour angle, degrees
        k_pir: conversion factor from radians to degrees
        rw: dimensionless variable substitute
        ru: dimensionless variable substitute
        rv: dimensionless variable substitute
        rnl: net longwave radiation, W/m^2

    Result:
        rn_d: daily net radiation, J/m^2
    """
    secs_d = 86400  # seconds in one solar day

    rn_d = (secs_d / np.pi) * (
        hn * k_pir * (rw * ru - rnl) + rw * rv * np.sin(np.deg2rad(hn))
    )

    return rn_d


def calc_nighttime_net_radiation(
    rw: NDArray,
    rv: NDArray,
    ru: NDArray,
    hs: NDArray,
    hn: NDArray,
    k_pir: float,
    rnl: NDArray,
) -> NDArray:
    """Calculates nightime net radiation, J/m^2.

    Args:
        rw: dimensionless variable substitute
        rv: dimensionless variable substitute
        ru: dimensionless variable substitute
        hs: sunset hour angle, degrees
        hn: crossover hour angle, degrees
        k_pir: conversion factor from radians to degrees
        rnl: net longwave radiation, rnl, W/m^2

    Returns:
        rnn_d: nighttime net radiation, J/m^2
    """
    secs_d = 86400  # seconds in one solar day

    rnn_d = (
        (rw * rv * (np.sin(np.deg2rad(hs)) - np.sin(np.deg2rad(hn))))
        + (rw * ru * k_pir * (hs - hn))
        - (rnl * (np.pi - k_pir * hn))
    ) * (secs_d / np.pi)

    return rnn_d


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
