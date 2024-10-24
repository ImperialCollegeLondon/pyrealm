"""The :mod:`~pyrealm.core.solar` submodule provides functions to calculate
photosynthetic photon flux density (ppfd), daily solar radiation fluxes and other
radiative values.
"""  # noqa: D205

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.calendar import LocationDateTime
from pyrealm.core.utilities import check_input_shapes


def calc_distance_factor(nu: NDArray, k_e: float) -> NDArray:
    r"""Calculates distance factor.

    This function calculates the distance factor :math:`dr` using the method of
    :cite:t:`berger:1993a`.

    .. math::

        dr = \left( 1.0 \mathbin{/}
               \left(\frac{1.0 - k_{e}^2}
                          {1.0 + k_{e} \cos\left(\nu \cdot \pi \mathbin{/}
                          180)\right)}
               \right)
             \right)^2

    Args:
        nu: Heliocentric true anomaly (degrees)
        k_e: Solar eccentricity

    Returns:
        An array of distance factors
    """

    dr = (1.0 / ((1.0 - k_e**2) / (1.0 + k_e * np.cos(np.deg2rad(nu))))) ** 2

    return dr


def calc_declination_angle_delta(
    lambda_: NDArray, k_eps: float, k_pir: float
) -> NDArray:
    r"""Calculates declination angle delta.

    This function calculates the solar declination angle delta using the method of
    :cite:t:`Woolf:1968`.

    .. math::

        \delta = \frac{\arcsin(\sin(\deg2rad(\lambda_))
            \cdot \sin(\deg2rad(k_{eps})))}{k_{pir}}

    Args:
        lambda_: heliocentric longitude
        k_eps: Solar obliquity
        k_pir: conversion factor from radians to degrees

    Returns:
        An array of solar declination angle delta
    """

    delta = np.arcsin(np.sin(np.deg2rad(lambda_)) * np.sin(np.deg2rad(k_eps))) / k_pir

    return delta


def calc_lat_delta_intermediates(
    delta: NDArray, latitude: NDArray
) -> tuple[NDArray, NDArray]:
    r"""Calculates intermediate values for use in solar radiation calcs.

    This function calculates :math:`ru` and :math:`rv` which are dimensionless
    intermediate values calculated from the solar declination angle delta and the
    observation latitude.

    .. math::

        ru = \sin(\deg2rad(\delta)) \cdot \sin(\deg2rad(\text{latitude}))

    .. math::

        rv = \cos(\deg2rad(\delta)) \cdot \cos(\deg2rad(\text{latitude}))

    Args:
        delta: solar declination delta
        latitude: observation latitude (degrees)

    Returns:
        A Tuple of :math:`ru` and :math:`rv`, calculation intermediates, unitless

    """
    ru = np.sin(np.deg2rad(delta)) * np.sin(np.deg2rad(latitude))
    rv = np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(latitude))

    return ru, rv


def calc_sunset_hour_angle(delta: NDArray, latitude: NDArray, k_pir: float) -> NDArray:
    r"""Calculates sunset hour angle.

    This function calculates the sunset hour angle :math:`hs` using eq3.22
        :cite:t:`stine_geyer:2001`.

    .. math::

        hs = \frac{\arccos(-1.0 \cdot \text{clip}(\frac{ru}{rv}, -1.0,
        1.0))}{k_{pir}}

    Args:
        delta: solar declination delta
        latitude: site latitude(s)
        k_pir: constant rad to degrees conversion, degrees/rad

    Returns:
        An array of local hour angle, degrees
    """

    ru, rv = calc_lat_delta_intermediates(delta=delta, latitude=latitude)

    return _calc_sunset_hour_angle_from_ru_rv(ru, rv, k_pir)


def _calc_sunset_hour_angle_from_ru_rv(
    ru: NDArray, rv: NDArray, k_pir: float
) -> NDArray:
    """Calculate sunset hour angle from intermediates.

    This function calculates the sunset hour angle using Eq3.22,
    :cite:t:'stine_geyer:2001'.

    Args:
        ru: dimensionless parameter
        rv: dimensionless parameter
        k_pir: constant rad to degrees conversion, degrees/rad

    Returns:
        An array of local hour angle, degrees
    """

    return np.arccos(-1.0 * np.clip(ru / rv, -1.0, 1.0)) / k_pir


def calc_daily_solar_radiation(
    dr: NDArray, hs: NDArray, delta: NDArray, latitude: NDArray, const: CoreConst
) -> NDArray:
    r"""Calculate daily extraterrestrial solar radiation (J/m^2).

    This function calculates the daily extraterrestrial solar radition :math:`J/m^2`
    using Eq. 1.10.3 :cite:t:`Duffie:2013a`.

    .. math::

        ra\_d = \left( \frac{secs_{d}}{\pi} \right) \cdot rad\_const \cdot dr
        \cdot \left(ru \cdot k_{pir} \cdot hs + rv \cdot
        \sin(\deg2rad(hs))\right)

    Args:
        rad_const: planetary radiation constant, W/m^2
        dr: dimensionless distance factor
        hs: local hour angle, degrees
        delta: solar declination delta
        latitude: site latitude(s)
        const: CoreConst object containing core constants:

    Returns:
        NDArray: An array of daily solar radiation, J/m^2
    """

    ru, rv = calc_lat_delta_intermediates(delta=delta, latitude=latitude)

    return _calc_daily_solar_radiation(dr=dr, ru=ru, rv=rv, hs=hs, const=const)


def _calc_daily_solar_radiation(
    dr: NDArray, ru: NDArray, rv: NDArray, hs: NDArray, const: CoreConst
) -> NDArray:
    """Calculate daily extraterrestrial solar radiation (J/m^2).

    This function calculates the daily extraterrestrial solar radition (J/m^2) using
    Eq. 1.10.3, :cite:t:`Duffie:2013a`.

    Args:
        dr: dimensionless distance factor
        ru: dimensionless variable substitute
        rv: dimensionless variable substitute
        hs: local hour angle, degrees
        const: CoreConst object

    Returns:
        An array of daily solar radiation, J/m^2
    """
    k_pir = const.k_pir
    k_secs_d = const.k_secs_d
    k_Gsc = const.k_Gsc

    ra_d = (
        (k_secs_d / np.pi)
        * k_Gsc
        * dr
        * (ru * k_pir * hs + rv * np.sin(np.deg2rad(hs)))
    )

    return ra_d


def calc_transmissivity(sf: NDArray, elv: NDArray, k_c: float, k_d: float) -> NDArray:
    r"""Calculate atmospheric transmissivity, :math:`tau`.

    This function calculates atmospheric transmissivity using the method of Eq.11,
    :cite:t:`Linacre:1968a` and Eq 2, :cite:t:`allen:1996a`.

    .. math::

        \tau = (k_{c} + k_{d} \cdot sf) \cdot (1.0 + (2.67 \times 10^{-5}) \cdot elv)

    Args:
        sf: Daily sunshine fraction of observations, unitless
        elv: Elevation of observations, metres
        k_c: dimensionless cloudy transmissivity
        k_d: dimensionless angular coefficient of transmissivity

    Returns:
        An array of bulk transmissivity, unitless
    """

    tau = (k_c + k_d * sf) * (1.0 + (2.67e-5) * elv)

    return tau


def calc_ppfd_from_tau_ra_d(
    tau: NDArray, ra_d: NDArray, k_fFEC: float, k_alb_vis: float
) -> NDArray:
    r"""Calculate photosynthetic photon flux density, :math:`PPFD`,(mol/m^2).

    This function calculates the :math:`PPFD` in :math:`mol/m^2` from secondary 
    calculated variables, and constants.

    .. math::

        ppfd = (1.0 \times 10^{-6}) \cdot k_{fFEC} \cdot (1.0 - k_{alb_vis})
        \cdot \tau \
            \cdot ra_{d}

    Args:
        tau: bulk transmissivity, unitless
        ra_d: daily solar radiation, J/m^2
        k_fFEC: from flux to energy conversion, umol/J
        k_alb_vis: visible light albedo

    Returns:
        An array of photosynthetic photon flux density, mol/m^2
    """

    ppfd = (1.0e-6) * k_fFEC * (1.0 - k_alb_vis) * tau * ra_d

    return ppfd


def calc_ppfd(
    sf: NDArray,
    elv: NDArray,
    latitude: NDArray,
    julian_day: NDArray,
    n_days: NDArray,
    const: CoreConst,
) -> NDArray:
    r"""Calculates Photosynthetic Photon Flux Density, :math:`PPFD`, (:math:`mol/m^2`).

    This function calculates :math:`PPFD` (:math:`mol/m^2`) from the observation 
    location and time using the following calculations:

    - :func:`calc_heliocentric_longitudes`
    - :func:`calc_distance_factor`
    - :func:`calc_declination_angle_delta`
    - :func:`calc_transmissivity`
    - :func:`calc_sunset_hour_angle`
    - :func:`calc_daily_solar_radiation`
    - :func:`calc_ppfd_from_tau_ra_d`

    Args:
        sf: Daily sunshine fraction of observations, unitless.
        elv: Elevation of observations, metres.
        latitude: The Latitude of observations (degrees).
        julian_day: Julian day of the year.
        n_days: Days in the calendar year.
        const: CoreConst object.

    Returns:
        An array of photosynthetic photon flux density, :math:`PPFD` (:math:`mol/m^2`).

    Example:
        >>> # Calculate ppfd
        >>> import numpy as np
        >>> from pyrealm.constants import CoreConst
        >>> # Create dataclass of constants
        >>> const = CoreConst()
        >>> # Define variable values
        >>> sf = np.array([1.0])
        >>> elv = np.array([142])
        >>> latitude = np.array([37.7])
        >>> julian_day = np.array([172])
        >>> n_days = np.array([366])
        >>> # Evaluate function
        >>> calc_ppfd(sf=sf, elv=elv, latitude=latitude, julian_day=julian_day, \
        ... n_days=n_days, const=const)
        array([62.04230021])
        
    """

    # Validate the inputs
    _ = check_input_shapes(latitude, elv, sf, julian_day, n_days)

    # Calculate intermediate values

    nu, lambda_ = calc_heliocentric_longitudes(julian_day=julian_day, n_days=n_days)

    dr = calc_distance_factor(nu=nu, k_e=const.k_e)

    delta = calc_declination_angle_delta(
        lambda_=lambda_, k_eps=const.k_eps, k_pir=const.k_pir
    )

    tau = calc_transmissivity(sf=sf, elv=elv, k_c=const.k_c, k_d=const.k_d)

    hs = calc_sunset_hour_angle(delta=delta, latitude=latitude, k_pir=const.k_pir)

    ra_d = calc_daily_solar_radiation(
        dr=dr, hs=hs, delta=delta, latitude=latitude, const=const
    )

    # Calculate ppfd

    ppfd = calc_ppfd_from_tau_ra_d(
        tau=tau, ra_d=ra_d, k_fFEC=const.k_fFEC, k_alb_vis=const.k_alb_vis
    )

    return ppfd


def calc_net_longwave_radiation(
    sf: NDArray, tc: NDArray, k_b: float, k_A: float
) -> NDArray:
    r"""Calculates net longwave radiation, :math:`rnl`, :math:`W/m^2`.

    This function calculates net longwave radiation in :math:`W/m^2` using the methods
    of Eq. 11, :cite:t:`colinprentice:1993a`, Eq. 5 and 6 :cite:t:`Linacre:1968a` .

    .. math::

        rnl = (k_{b} + (1.0 - k_{b}) \cdot sf) \cdot (k_{A} - tc)

    Args:
        sf: sunshine fraction of observations, unitless
        tc: temperature of observations, °C
        k_b: calculation constant for Rnl
        k_A: calculation constant for Rnl

    Returns:
        An array of net longwave radiation, :math:`W/m^2`.
    """

    rnl = (k_b + (1.0 - k_b) * sf) * (k_A - tc)

    return rnl


def calc_rw(tau: NDArray, dr: NDArray, k_alb_sw: float, k_Gsc: float) -> NDArray:
    r"""Calculates variable substitute :math:`rw`, :math:`W/m^2`.

    .. math::

        rw = (1.0 - k_{alb_sw}) \cdot \tau \cdot k_{Gsc} \cdot dr

    Args:
        tau          : bulk transmissivity, unitless
        dr           : distance ration, unitless
        k_alb_sw     : shortwave albedo
        k_Gsc        : solar constant, W/m^2

    Returns:
        An array of intermediate variable rw, :math:`W/m^2`.
    """

    rw = (1.0 - k_alb_sw) * tau * k_Gsc * dr

    return rw


def calc_net_rad_crossover_hour_angle(
    rnl: NDArray,
    tau: NDArray,
    dr: NDArray,
    delta: NDArray,
    latitude: NDArray,
    const: CoreConst,
) -> NDArray:
    r"""Calculates the net radiation crossover hour angle, :math:`hn` degrees.

    This function calculates the net radiation crossover hour angle :math:`hn` in
    degrees.

    .. math::

        hn = \frac{\arccos(\text{clip}((rnl - rw \cdot ru) / (rw \cdot rv), -1.0, 1.0))}
        {k_{pir}}

    Args:
        rnl: net longwave radiation, :math:`W/m^2`
        tau: bulk transmissivity, unitless
        dr: distance ration, unitless
        delta: solar declination delta
        latitude: site latitude(s)
        const: CoreConst object

    Returns:
        _calc_net_rad_crossover_hour_angle
    """

    ru, rv = calc_lat_delta_intermediates(delta=delta, latitude=latitude)
    rw = calc_rw(tau=tau, dr=dr, k_alb_sw=const.k_alb_sw, k_Gsc=const.k_Gsc)

    return _calc_net_rad_crossover_hour_angle(
        rnl=rnl, rw=rw, ru=ru, rv=rv, k_pir=const.k_pir
    )


def _calc_net_rad_crossover_hour_angle(
    rnl: NDArray, rw: NDArray, ru: NDArray, rv: NDArray, k_pir: float
) -> NDArray:
    """Calculates the net radiation crossover hour angle, degrees.

    This function calculates the net radiation crossover hour angle in degrees.

    Args:
        rnl: net longwave radiation, W/m^2
        rw: intermediate variable, W/m^2
        ru: intermediate variable, W/m^2
        rv: intermediate variable, W/m^2
        k_pir: conversion factor from radians to degrees

    Returns:
        An array of net radiation crossover hour angle, degrees
    """

    hn = np.arccos(np.clip((rnl - rw * ru) / (rw * rv), -1.0, 1.0)) / k_pir

    return hn


def calc_daytime_net_radiation(
    hn: NDArray,
    rnl: NDArray,
    delta: NDArray,
    latitude: NDArray,
    tau: NDArray,
    dr: NDArray,
    const: CoreConst,
) -> NDArray:
    r"""Calculates daily net radiation, :math:`rn_{d}`, :math:`J/m^2`.

    .. math::

        rn_{d} = \left( \frac{secs_{d}}{\pi} \right) \cdot \
        \left( hn \cdot k_{pir} \cdot (rw \cdot ru - rnl) + rw \cdot rv \cdot \
        sin(\deg2rad(hn)) \right)

    Args:
        hn: crossover hour angle, degrees
        rnl: net longwave radiation, W/m^2
        delta: solar declination delta
        latitude: site latitude(s)
        tau: bulk transmissivity, unitless
        dr: distance ration, unitless
        const: CoreConst object

    Result:
        _calc_daytime_net_radiation
    """
    ru, rv = calc_lat_delta_intermediates(delta=delta, latitude=latitude)
    rw = calc_rw(tau=tau, dr=dr, k_alb_sw=const.k_alb_sw, k_Gsc=const.k_Gsc)

    return _calc_daytime_net_radiation(
        hn=hn, rw=rw, ru=ru, rv=rv, rnl=rnl, k_pir=const.k_pir, k_secs_d=const.k_secs_d
    )


def _calc_daytime_net_radiation(
    hn: NDArray,
    rw: NDArray,
    ru: NDArray,
    rv: NDArray,
    rnl: NDArray,
    k_pir: float,
    k_secs_d: int,
) -> NDArray:
    """Calculates daily net radiation, :math:`J/m^2`.

    Args:
        hn: crossover hour angle, degrees
        rw: dimensionless variable substitute
        ru: dimensionless variable substitute
        rv: dimensionless variable substitute
        rnl: net longwave radiation, W/m^2
        k_pir: conversion factor from radians to degrees
        k_secs_d: seconds in one solar day

    Result:
        An array of daily net radiation, J/m^2
    """

    rn_d = (k_secs_d / np.pi) * (
        hn * k_pir * (rw * ru - rnl) + rw * rv * np.sin(np.deg2rad(hn))
    )

    return rn_d


def calc_nighttime_net_radiation(
    rnl: NDArray,
    hn: NDArray,
    hs: NDArray,
    delta: NDArray,
    latitude: NDArray,
    tau: NDArray,
    dr: NDArray,
    const: CoreConst,
) -> NDArray:
    r"""Calculates nightime net radiation, :math:`rnn_{d}` :math:`J/m^2`.

    .. math::

        rnn_{d} = \left( 
        rw \cdot rv \cdot (\sin(\deg2rad(hs)) - \sin(\deg2rad(hn))) \
        + rw \cdot ru \cdot k_{pir} \cdot (hs - hn)
        - rnl \cdot (\pi - k_{pir} \cdot hn)
        \right) \cdot \left( \frac{secs_{d}}{\pi} \right)

    Args:
        rnl: net longwave radiation, rnl, :math:`W/m^2`
        hs: sunset hour angle, degrees
        hn: crossover hour angle, degrees
        delta: solar declination delta
        latitude: site latitude(s)
        tau: bulk transmissivity, unitless
        dr: distance ration, unitless
        const: CoreConst object

    Returns:
        _calc_nighttime_net_radiation
    """
    ru, rv = calc_lat_delta_intermediates(delta=delta, latitude=latitude)
    rw = calc_rw(tau=tau, dr=dr, k_alb_sw=const.k_alb_sw, k_Gsc=const.k_Gsc)

    return _calc_nighttime_net_radiation(
        rw=rw,
        rv=rv,
        ru=ru,
        hs=hs,
        hn=hn,
        rnl=rnl,
        k_pir=const.k_pir,
        k_secs_d=const.k_secs_d,
    )


def _calc_nighttime_net_radiation(
    rw: NDArray,
    rv: NDArray,
    ru: NDArray,
    hs: NDArray,
    hn: NDArray,
    rnl: NDArray,
    k_pir: float,
    k_secs_d: int,
) -> NDArray:
    """Calculates nightime net radiation, :math:`J/m^2`.

    Args:
        rw: dimensionless variable substitute
        rv: dimensionless variable substitute
        ru: dimensionless variable substitute
        hs: sunset hour angle, degrees
        hn: crossover hour angle, degrees
        rnl: net longwave radiation, rnl, :math:`W/m^2`
        k_pir: conversion factor from radians to degrees
        k_secs_d: seconds in one solar day

    Returns:
        An array of nighttime net radiation, :math:`J/m^2`
    """

    rnn_d = (
        (rw * rv * (np.sin(np.deg2rad(hs)) - np.sin(np.deg2rad(hn))))
        + (rw * ru * k_pir * (hs - hn))
        - (rnl * (np.pi - k_pir * hn))
    ) * (k_secs_d / np.pi)

    return rnn_d


def calc_heliocentric_longitudes(
    julian_day: NDArray[np.float64],
    n_days: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> tuple[NDArray, NDArray]:
    """Calculate heliocentric longitude and anomaly.

    This function calculates the heliocentric true anomaly (``nu``, degrees) and true
    longitude (``lambda_``, degrees), given the Julian day in the year and the
    number of days in the year, following :cite:t:`berger:1978a`.

    Args:
        julian_day: day of year
        n_days: number of days in year
        core_const: An instance of CoreConst.

    Returns:
        A tuple of NDArrays containing ``nu`` and ``lambda_``.
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


def calc_solar_elevation(site_obs_data: LocationDateTime) -> NDArray:
    r"""Calculate the solar elevation angle for a specific location and times.

    This function calculates the solar elevation angle, which is the angle between the 
    sun and the observer's local horizon, using the methods outlined in 
    :cite:t:`depury:1997a`.

    NB: This implementation does not correct for the effect of local observation 
    altitude on perceived solar elevation.

    This approach uses the following calculations:

        - :func:`day_angle`
        - :func:`equation_of_time`
        - :func:`solar_noon`
        - :func:`local_hour_angle`
        - :func:`solar_declination`
        - :func:`elevation_from_lat_dec_hn`

    Args:
        site_obs_data: A :class:`~pyrealm.core.calendar.LocationDateTime` instance 
            containing the location and time-specific information for the observation 
            site.

    Returns:
        An array of solar elevation angles in radians, representing the angular height 
        of the sun above the horizon at the specified location and time.

    Example:
        >>> # Calculate solar elevation at Wagga Wagga, Australia on 25th October 1995.
        >>> # Worked example taken from dePury and Farquhar (1997)
        >>> import numpy as np
        >>> from pyrealm.core.calendar import LocationDateTime
        >>> from pyrealm.core.solar import calc_solar_elevation
        >>> # Create instance of LocationDateTime dataclass
        >>> latitude = -35.058333
        >>> longitude = 147.34167
        >>> year_date_time = np.array([np.datetime64("1995-10-25T10:30")])
        >>> ldt = LocationDateTime(latitude = latitude, longitude = longitude,\
        ...     year_date_time = year_date_time)
        >>> # Run solar elevation calculation
        >>> calc_solar_elevation(ldt)
        array([1.0615713])

    """

    G_d = day_angle(site_obs_data.julian_days)

    E_t = equation_of_time(G_d)

    t0 = solar_noon(site_obs_data.longitude, site_obs_data.local_standard_meridian, E_t)

    hour_angle = local_hour_angle(site_obs_data.decimal_time, t0)

    declination = solar_declination(site_obs_data.julian_days)

    elevation = elevation_from_lat_dec_hn(
        site_obs_data.latitude_rad, declination, hour_angle
    )
    return elevation


def elevation_from_lat_dec_hn(
    latitude: NDArray | float, declination: NDArray, hour_angle: NDArray
) -> NDArray:
    r"""Calculate the elevation angle of the sun above the horizon.

    The elevation angle (or solar altitude angle) is the angle between the horizon and 
    the sun, which indicates how high the sun is in the sky at a given time. This 
    function calculates the elevation angle based on the observer's latitude, the 
    solar declination, and the hour angle.

    The calculation is based on the following trigonometric relationship based on Eqn 
    A13, :cite:t:`depury:1997a`:

    .. math::
        \sin(\alpha) = \sin(\phi) \cdot \sin(\delta) +
        \cos(\phi) \cdot \cos(\delta) \cdot \cos(h)

    where,

    - :math:`\alpha` is the elevation angle,
    - :math:`\phi` is the latitude of the observer,
    - :math:`\delta` is the solar declination, and
    - :math:`h` is the hour angle.

    The elevation angle is then given by:

    .. math::
        \alpha = \arcsin(\sin(\alpha))

    Args:
        latitude: Array of latitudes in radians, or a single latitude value (as a \
            float).
        declination: Array of solar declination angles in radians.
        hour_angle: Array of hour angles in radians.

    Returns:
        An array of elevation angles in radians (as a floating-point number array),
        representing the angular height of the sun above the horizon.

    """

    sin_alpha = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(
        declination
    ) * np.cos(hour_angle)

    elevation = np.arcsin(sin_alpha)

    return elevation


def solar_declination(td: NDArray) -> NDArray:
    r"""Calculates solar declination angle.

    Use method described in eqn A14 of :cite:t:`depury:1997a` to calculate solar
    declination angle, from day of the year.

    .. math::

        \text{declination} = -23.4 \cdot \left(\frac{1}{k\_pir}\right) \cdot \cos\left
        (\frac{2 \cdot \pi \cdot (td + 10)}{365}\right)

    Args:
        td: Julian day of the year

    Returns:
        Array of solar declination angles (radians)
    """

    declination = -23.4 * (np.pi / 180) * np.cos((2 * np.pi * (td + 10)) / 365)

    return declination


def local_hour_angle(t: NDArray, t0: NDArray) -> NDArray:
    r"""Calculate the local hour angle :math:`h` for a given time and solar noon.

    The local hour angle is a measure of time, expressed in angular terms, that
    indicates the position of the sun relative to solar noon. This function
    calculates the local hour angle by determining the difference between the
    current time (``t``) and the solar noon time (:math:`t_{0}`), and then
    converting this difference into an angle.

    Equation implemented from A15 :cite:t:`depury:1997a`.

    .. math::
        h = \pi \cdot \frac{t - t_{0}}{12}

    Args:
        t: Array of current time values in hours (as a floating-point number).
        t0: Array of solar noon time values in hours (as a floating-point number).

    Returns:
        The local hour angle in radians (as a floating-point number array), which
        represents the angular distance of the sun from the local meridian at the
        given time.

    """

    h = np.pi * (t - t0) / 12

    return h


def solar_noon(L_e: float, L_s: float, E_t: NDArray) -> NDArray:
    r"""Calculate the solar noon time for a given location.

    The solar noon is the time of day when the sun is at its highest point in the sky 
    for a given location. This function calculates the solar noon by adjusting the 
    standard noon time (12:00 PM) based on the difference between the local 
    longitude (:math:`L_{e}`) and the local standard meridian (:math:`L_{s}`) and 
    the equation of time (:math:`E_{t}`). Based on EqA16, :cite:t:`depury:1997a`.

    .. math:: t_{0} = 12 + \frac{4 \cdot -(L_{e} - L_{s}) - E_{t}}{60}

    Args:
        L_e: Local longitude of the observer in degrees (positive for east,negative \
            for west).
        L_s: Longitude of the standard meridian for the observer's time zone in degrees.
        E_t: Equation of time in minutes, accounting for the irregularity of the \
            Earth's orbit and axial tilt.

    Returns:
        The solar noon time in hours (as a floating-point number), which can be
        interpreted as a time of day.

    """

    t0 = 12 + (4 * -(L_e - L_s) - E_t) / 60

    return t0


def equation_of_time(day_angle: NDArray) -> NDArray:
    r"""Calculates equation of time in minutes.

    Based on eqn 1.4.1 :cite:t:`iqbal:1983a` rather than eqn A17
    :cite:t:`depury:1997a` due to incorrect reported implementation in the latter.

    .. math::

        E_t = \left( 0.000075
            + 0.001868 \cdot \cos(\Gamma)
            - 0.032077 \cdot \sin(\Gamma)
            - 0.014615 \cdot \cos(2\Gamma)
            - 0.04089 \cdot \sin(2\Gamma) \right)
            \times 229.18

    Where gamma is the day angle.

    Args:
        day_angle: day angle in radians

    Returns:
        An array of Equation of time values

    """
    E_t = (
        0.000075
        + 0.001868 * np.cos(day_angle)
        - 0.032077 * np.sin(day_angle)
        - 0.014615 * np.cos(2 * day_angle)
        - 0.04089 * np.sin(2 * day_angle)
    ) * 229.18

    return E_t


def day_angle(t_d: NDArray) -> NDArray:
    r"""Calculates solar day angle (gamma), radians.

    The day angle (``gamma``) for a given day of the year ``N``, (where N=1 for
    January 1st and N=365 for December 31st) can be calculated using the following
    formula:

    Based on Eqn A18, :cite:t:`depury:1997a`.

    .. math::

    \gamma = \frac{2\pi (N - 1)}{365}

    Args:
        t_d: Julian day of the year

    Returns:
        An array of solar day angles

    """

    day_angle = 2 * np.pi * (t_d - 1) / 365

    return day_angle
