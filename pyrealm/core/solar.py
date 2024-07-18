"""The :mod:`~pyrealm.core.solar` submodule provides functions to calculate
photosynthetic photon flux density (ppfd), daily solar radiation fluxes and other
radiative values.
"""  # noqa: D205

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.utilities import check_input_shapes


def calc_distance_factor(nu: NDArray, k_e: float) -> NDArray:
    r"""Calculates distance factor.

    This function calculates distance factor using the method of Berger et al. (1993)

    .. math::

        dr = \left( \frac{1.0}{\left(\frac{1.0 - k_e^2}{1.0 + k_e \cos\left(\deg2rad(\nu)\right)}\right)} \right)^2

    Args:
        nu          : heliocentric true anomaly (degrees)
        k_e         : Solar eccentricity

    Returns:
        dr          : distance factor
    """  # noqa: E501

    dr = (1.0 / ((1.0 - k_e**2) / (1.0 + k_e * np.cos(np.deg2rad(nu))))) ** 2

    return dr


def calc_declination_angle_delta(
    lambda_: NDArray, k_eps: float, k_pir: float
) -> NDArray:
    r"""Calculates declination angle delta.

    This function calculates the solar declination angle delta using
    the method of Woolf (1968)

    .. math::

        \delta = \frac{\arcsin(\sin(\deg2rad(\lambda)) \cdot \sin(\deg2rad(k\_eps)))}{k\_pir}

    Args:
        lambda_     : heliocentric longitude
        k_eps       : Solar obliquity
        k_pir       : conversion factor from radians to degrees

    Returns:
        delta       : solar declination angle delta
    """  # noqa: E501
    delta = np.arcsin(np.sin(np.deg2rad(lambda_)) * np.sin(np.deg2rad(k_eps))) / k_pir

    return delta


def calc_lat_delta_intermediates(
    delta: NDArray, latitude: NDArray
) -> tuple[NDArray, NDArray]:
    r"""Calculates intermediate values for use in solar radiation calcs.

    This function calculates ru and rv which are dimensionless intermediate
    values calculated from the solar declination angle delta and the
    observation latitude.

    .. math::

        ru = \sin(\deg2rad(\delta)) \cdot \sin(\deg2rad(\text{latitude}))

    .. math::

        rv = \cos(\deg2rad(\delta)) \cdot \cos(\deg2rad(\text{latitude}))

    Args:
        delta       : solar declination delta
        latitude    : observation latitude

    Returns:
        ru, rv      : calculation intermediates, unitless

    """
    ru = np.sin(np.deg2rad(delta)) * np.sin(np.deg2rad(latitude))
    rv = np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(latitude))

    return ru, rv


def calc_sunset_hour_angle(delta: NDArray, latitude: NDArray, k_pir: float) -> NDArray:
    r"""Calculates sunset hour angle.

    This function calculates the sunset hour angle using Eq3.22, Stine & Geyer (2001)

    .. math::

        hs = \frac{\arccos(-1.0 \cdot \text{clip}(\frac{ru}{rv}, -1.0, 1.0))}{k\_pir}

    Args:
        delta       : solar declination delta
        latitude    : site latitude(s)
        k_pir       : constant rad to degrees conversion, degrees/rad

    Returns:
        hour angle  : local hour angle, degrees
    """
    ru, rv = calc_lat_delta_intermediates(delta, latitude)

    return _calc_sunset_hour_angle_from_ru_rv(ru, rv, k_pir)


def _calc_sunset_hour_angle_from_ru_rv(
    ru: NDArray, rv: NDArray, k_pir: float
) -> NDArray:
    """Calculate sunset hour angle from intermediates.

    This function calculates the sunset hour angle using Eq3.22, Stine & Geyer (2001).

    Args:
        ru          : dimensionless parameter
        rv          : dimensionless parameter
        k_pir       : constant rad to degrees conversion, degrees/rad
    Returns:
        hour angle  : local hour angle, degrees
    """

    return np.arccos(-1.0 * np.clip(ru / rv, -1.0, 1.0)) / k_pir


def calc_daily_solar_radiation(
    rad_const: float,
    dr: NDArray,
    k_pir: float,
    hs: NDArray,
    delta: NDArray,
    latitude: NDArray,
) -> NDArray:
    r"""Calculate daily extraterrestrial solar radiation (J/m^2).

    This function calculates the daily extraterrestrial solar radition (J/m^2)
    using Eq. 1.10.3, Duffy & Beckman (1993)

    .. math::

        ra\_d = \left( \frac{secs\_d}{\pi} \right) \cdot rad\_const \cdot dr \cdot
        \left(ru \cdot k\_pir \cdot hs + rv \cdot \sin(\deg2rad(hs))\right)

    Args:
        rad_const   : planetary radiation constant, W/m^2
        dr          : dimensionless distance factor
        k_pir       : radians to degrees conversion, degrees/rad
        hs          : local hour angle, degrees
        delta       : solar declination delta
        latitude    : site latitude(s)


    Returns:
        ra_d        : daily solar radiation, J/m^2
    """
    ru, rv = calc_lat_delta_intermediates(delta, latitude)

    return _calc_daily_solar_radiation(rad_const, dr, ru, rv, k_pir, hs)


def _calc_daily_solar_radiation(
    rad_const: float, dr: NDArray, ru: NDArray, rv: NDArray, k_pir: float, hs: NDArray
) -> NDArray:
    """Calculate daily extraterrestrial solar radiation (J/m^2).

    This function calculates the daily extraterrestrial solar radition (J/m^2)
    using Eq. 1.10.3, Duffy & Beckman (1993)

    Args:
        rad_const   : planetary radiation constant, W/m^2
        dr          : dimensionless distance factor
        ru          : dimensionless variable substitute
        rv          : dimensionless variable substitute
        k_pir       : radians to degrees conversion, degrees/rad
        hs          : local hour angle, degrees

    Returns:
        ra_d        : daily solar radiation, J/m^2
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
    r"""Calculate atmospheric transmissivity, tau.

    This function calculates atmospheric transmissivity using the method of
    Eq.11, Linacre (1968) and Eq 2, Allen (1996)

    .. math::

        \tau = (k\_c + k\_d \cdot sf) \cdot (1.0 + (2.67 \times 10^{-5}) \cdot elv)

    Args:
        k_c         : dimensionless cloudy transmissivity
        k_d         : dimensionless angular coefficient of transmissivity
        sf          : Daily sunshine fraction of observations, unitless
        elv         : Elevation of observations, metres

    Returns:
        tau         : bulk transmissivity, unitless
    """
    tau = (k_c + k_d * sf) * (1.0 + (2.67e-5) * elv)
    return tau


def calc_ppfd_from_tau_ra_d(
    k_fFEC: float, k_alb_vis: float, tau: NDArray, ra_d: NDArray
) -> NDArray:
    r"""Calculate photosynthetic photon flux density, PPFD,(mol/m^2).

    This function calculates the PPFD in mol/m^2 from secondary calculated
    variables and constants.

    .. math::

        ppfd = (1.0 \times 10^{-6}) \cdot k\_fFEC \cdot (1.0 - k\_alb\_vis) \cdot \tau \
            cdot ra\_d

    Args:
        k_fFEC      : from flux to energy conversion, umol/J
        k_alb_vis   : visible light albedo
        tau         : bulk transmissivity, unitless
        ra_d        : daily solar radiation, J/m^2

    Returns:
        ppfd        : photosynthetic photon flux density, mol/m^2
    """

    ppfd = (1.0e-6) * k_fFEC * (1.0 - k_alb_vis) * tau * ra_d

    return ppfd


def calc_ppfd(
    sf: NDArray,
    elv: NDArray,
    latitude: NDArray,
    julian_day: NDArray,
    n_days: NDArray,
    k_e: float,
    k_eps: float,
    k_pir: float,
    k_d: float,
    k_Gsc: float,
    k_fFec: float,
    k_alb_vis: float,
    k_c: float,
) -> NDArray:
    """Calculates photosynthetic photon flux density, PPFD,(mol/m^2).

    This function calulates ppfd (mol/m^2) from primary variables.

    Args:
        sf          : Daily sunshine fraction of observations, unitless
        elv         : Elevation of observations, metres
        latitude    : The Latitude of observations (degrees)
        julian_day  : Julian day of the year
        n_days      : days in the calendar year
        k_e         : Solar eccentricity
        k_eps       : Solar obliquity
        k_pir       : Conversion factor from radians to degrees
        k_d         : angular coefficient of transmittivity
        k_Gsc       : solar constant, W/m^2
        k_fFec      : from flux to energy conversion, umol/J
        k_alb_vis   : visible light albedo
        k_c         : cloudy transmittivity

    Returns:
        ppfd        : photosynthetic photon flux density, PPFD,(mol/m^2)

    Example:
        >>> # Calculate ppfd
        >>> import numpy as np
        >>> from pyrealm.constants import CoreConst
        >>> # Create dataclass of constants
        >>> const = CoreConst()
        >>> # define variable values
        >>> sf = np.array([1.0])
        >>> elv = np.array([142])
        >>> latitide = np.array([37.7])
        >>> julian_day = np.array([172])
        >>> n_days = np.array([366])
        >>> # evaluate function
        >>> calc_ppfd(sf, elv, latitude, julian_day, n_days, const.k_e, const.k_eps,
        const.k_pir, const.k_d,const.k_Gsc, const.k_fFec, const.k_alb_vis, const.k_c)
        array([62.042300])


    """

    # Validate the inputs
    _ = check_input_shapes(latitude, elv, sf, julian_day, n_days)

    # Calculate intermediate values

    nu, lambda_ = calc_heliocentric_longitudes(julian_day, n_days)

    dr = calc_distance_factor(nu, k_e)

    delta = calc_declination_angle_delta(lambda_, k_eps, k_pir)

    tau = calc_transmissivity(k_c, k_d, sf, elv)

    hs = calc_sunset_hour_angle(delta, latitude, k_pir)

    ra_d = calc_daily_solar_radiation(k_Gsc, dr, k_pir, hs, delta, latitude)

    # Calculate ppfd

    ppfd = calc_ppfd_from_tau_ra_d(k_fFec, k_alb_vis, tau, ra_d)

    return ppfd


def calc_rnl(k_b: float, sf: NDArray, k_A: float, tc: NDArray) -> NDArray:
    r"""Calculates net longwave radiation, rnl, W/m^2.

    This function calculates net longwave radiation in W/m^2.

    .. math::

        rnl = (k\_b + (1.0 - k\_b) \cdot sf) \cdot (k\_A - tc)

    Args:
        k_b         : calculation constant for Rnl
        sf          : sunshine fraction of observations, unitless
        k_A         : calculation constant for Rnl
        tc          : temperature of observations, °C

    Returns:
        rnl         : net longwave radiation, W/m^2
    """

    rnl = (k_b + (1.0 - k_b) * sf) * (k_A - tc)

    return rnl


def calc_rw(k_alb_sw: float, tau: NDArray, k_Gsc: float, dr: NDArray) -> NDArray:
    r"""Calculates variable substitute rw, W/m^2.

    .. math::

        rw = (1.0 - k\_alb\_sw) \cdot \tau \cdot k\_Gsc \cdot dr

    Args:
        k_alb_sw    : shortwave albedo
        tau         : bulk transmissivity, unitless
        k_Gsc       : solar constant, W/m^2
        dr          : distance ration, unitless

    Returns:
        rw          : intermediate variable, W/m^2
    """

    rw = (1.0 - k_alb_sw) * tau * k_Gsc * dr

    return rw


def calc_net_rad_crossover_hour_angle(
    rnl: NDArray,
    k_pir: float,
    k_alb_sw: float,
    tau: NDArray,
    k_Gsc: float,
    dr: NDArray,
    delta: NDArray,
    latitude: NDArray,
) -> NDArray:
    r"""Calculates the net radiation crossover hour angle, degrees.

    This function calculates the net radiation crossover hour angle in degrees.

    .. math::

        hn = \frac{\arccos(\text{clip}((rnl - rw \cdot ru) / (rw \cdot rv), -1.0, 1.0))}
        {k\_pir}

    Args:
        rnl         : net longwave radiation, W/m^2
        k_pir       : conversion factor from radians to degrees
        k_alb_sw    : shortwave albedo
        tau         : bulk transmissivity, unitless
        k_Gsc       : solar constant, W/m^2
        dr          : distance ration, unitless
        delta       : solar declination delta
        latitude    : site latitude(s)


    Returns:
        _calc_net_rad_crossover_hour_angle
    """

    ru, rv = calc_lat_delta_intermediates(delta, latitude)
    rw = calc_rw(k_alb_sw, tau, k_Gsc, dr)

    return _calc_net_rad_crossover_hour_angle(rnl, rw, ru, rv, k_pir)


def _calc_net_rad_crossover_hour_angle(
    rnl: NDArray, rw: NDArray, ru: NDArray, rv: NDArray, k_pir: float
) -> NDArray:
    """Calculates the net radiation crossover hour angle, degrees.

    This function calculates the net radiation crossover hour angle in degrees.

    Args:
        rnl         : net longwave radiation, W/m^2
        rw          : intermediate variable, W/m^2
        ru          : intermediate variable, W/m^2
        rv          : intermediate variable, W/m^2
        k_pir       : conversion factor from radians to degrees

    Returns:
        hn          : net radiation crossover hour angle, degrees
    """

    hn = np.arccos(np.clip((rnl - rw * ru) / (rw * rv), -1.0, 1.0)) / k_pir

    return hn


def calc_daytime_net_radiation(
    hn: NDArray,
    k_pir: float,
    rnl: NDArray,
    delta: NDArray,
    latitude: NDArray,
    k_alb_sw: float,
    tau: NDArray,
    k_Gsc: float,
    dr: NDArray,
) -> NDArray:
    r"""Calculates daily net radiation, J/m^2.

    .. math::
        secs\_d = 86400  \quad \text{ \# seconds in one solar day}

        rn\_d = \left( \frac{secs\_d}{\pi} \right) \cdot 
        \left( hn \cdot k\_pir \cdot (rw \cdot ru - rnl) + rw \cdot rv \cdot \
        sin(\deg2rad(hn)) \right)

    Args:
        hn          : crossover hour angle, degrees
        k_pir       : conversion factor from radians to degrees
        rnl         : net longwave radiation, W/m^2
        delta       : solar declination delta
        latitude    : site latitude(s)
        k_alb_sw    : shortwave albedo
        tau         : bulk transmissivity, unitless
        k_Gsc       : solar constant, W/m^2
        dr          : distance ration, unitless

    Result:
        _calc_daytime_net_radiation
    """
    ru, rv = calc_lat_delta_intermediates(delta, latitude)
    rw = calc_rw(k_alb_sw, tau, k_Gsc, dr)

    return _calc_daytime_net_radiation(hn, k_pir, rw, ru, rv, rnl)


def _calc_daytime_net_radiation(
    hn: NDArray, k_pir: float, rw: NDArray, ru: NDArray, rv: NDArray, rnl: NDArray
) -> NDArray:
    """Calculates daily net radiation, J/m^2.

    Args:
        hn          : crossover hour angle, degrees
        k_pir       : conversion factor from radians to degrees
        rw          : dimensionless variable substitute
        ru          : dimensionless variable substitute
        rv          : dimensionless variable substitute
        rnl         : net longwave radiation, W/m^2

    Result:
        rn_d        : daily net radiation, J/m^2
    """

    secs_d = 86400  # seconds in one solar day

    rn_d = (secs_d / np.pi) * (
        hn * k_pir * (rw * ru - rnl) + rw * rv * np.sin(np.deg2rad(hn))
    )

    return rn_d


def calc_nighttime_net_radiation(
    k_pir: float,
    rnl: NDArray,
    hn: NDArray,
    hs: NDArray,
    delta: NDArray,
    latitude: NDArray,
    k_alb_sw: float,
    tau: NDArray,
    k_Gsc: float,
    dr: NDArray,
) -> NDArray:
    r"""Calculates nightime net radiation, J/m^2.

    .. math::

        secs\_d = 86400  \quad \text{ \# seconds in one solar day}

        rnn\_d = \left(
        rw \cdot rv \cdot (\sin(\deg2rad(hs)) - \sin(\deg2rad(hn)))
        + rw \cdot ru \cdot k\_pir \cdot (hs - hn)
        - rnl \cdot (\pi - k\_pir \cdot hn)
        \right) \cdot \left( \frac{secs\_d}{\pi} \right)

    Args:
        k_pir       : conversion factor from radians to degrees
        rnl         : net longwave radiation, rnl, W/m^2
        hs          : sunset hour angle, degrees
        hn          : crossover hour angle, degrees
        delta       : solar declination delta
        latitude    : site latitude(s)
        k_alb_sw    : shortwave albedo
        tau         : bulk transmissivity, unitless
        k_Gsc       : solar constant, W/m^2
        dr          : distance ration, unitless

    Returns:
        _calc_nighttime_net_radiation
    """
    ru, rv = calc_lat_delta_intermediates(delta, latitude)
    rw = calc_rw(k_alb_sw, tau, k_Gsc, dr)

    return _calc_nighttime_net_radiation(rw, rv, ru, hs, hn, k_pir, rnl)


def _calc_nighttime_net_radiation(
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
        rw          : dimensionless variable substitute
        rv          : dimensionless variable substitute
        ru          : dimensionless variable substitute
        hs          : sunset hour angle, degrees
        hn          : crossover hour angle, degrees
        k_pir       : conversion factor from radians to degrees
        rnl         : net longwave radiation, rnl, W/m^2

    Returns:
        rnn_d       : nighttime net radiation, J/m^2
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
        julian_day  : day of year
        n_days      : number of days in year
        core_const  : An instance of CoreConst.

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


def beta_angle_from_lat_dec_hour(
    latitude: NDArray, declination: NDArray, hour_angle: NDArray
) -> NDArray:
    """Calculates solar beta angle (elevation angle).

    Calculates solar beta angle using Eq A13 of dePury & Farquhar (1997).

    Args:
        latitude: array of latitudes (rads)
        declination: array of declinations (rads)
        hour_angle: array of hour angle (rads)

    Returns:
        beta: array of solar beta angles
    """

    beta = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(
        declination
    ) * np.cos(hour_angle)

    return beta
