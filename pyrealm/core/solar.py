"""The :mod:`~pyrealm.core.solar` submodule provides functions to calculate
photosynthetic photon flux density (PPFD), daily solar radiation fluxes and other
radiative values.
"""  # noqa: D205

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.calendar import LocationDateTime
from pyrealm.core.utilities import check_input_shapes


def calculate_distance_factor(
    nu: NDArray[np.float64], solar_eccentricity: float = CoreConst().solar_eccentricity
) -> NDArray[np.float64]:
    r"""Calculates distance factor.

    This function calculates the distance factor :math:`d_r` using the method of
    :cite:t:`berger:1993a`.

    .. math::

        d_r = \left( 1.0 \mathbin{/}
               \left(\frac{1.0 - e^2}
                          {1.0 + e \cos\left(\nu) \right)}
               \right)
             \right)^2

    Args:
        nu: Heliocentric true anomaly (:math:`\nu`, degrees)
        solar_eccentricity: Solar eccentricity (:math:`e`)

    Returns:
        An array of distance factors
    """

    return (
        1.0
        / (
            (1.0 - solar_eccentricity**2)
            / (1.0 + solar_eccentricity * np.cos(np.deg2rad(nu)))
        )
    ) ** 2


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


def calculate_ru_rv_intermediates(
    declination: NDArray[np.float64], latitude: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculates intermediate values for use in solar radiation calcs.

    This function calculates :math:`r_u` and :math:`r_v` which are dimensionless
    intermediate values calculated from the solar declination angle (:math:`\delta`) and
    the observation latitude (:math:`\phi`).

    .. math::
        :nowrap:

        \[
            \begin{align*}
                r_u &= \sin(\delta) \sin(\phi) \\
                r_v &= \cos(\delta) \cos(\phi)
            \end{align*}
        \]

    Args:
        declination: Solar declination (:math:`\delta`, degrees)
        latitude: Observation latitude (:math:`\phi`, degrees)

    Returns:
        A tuple of :math:`r_u` and :math:`r_v`.
    """
    ru = np.sin(np.deg2rad(declination)) * np.sin(np.deg2rad(latitude))
    rv = np.cos(np.deg2rad(declination)) * np.cos(np.deg2rad(latitude))

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

    ru, rv = calculate_ru_rv_intermediates(declination=delta, latitude=latitude)

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

    ru, rv = calculate_ru_rv_intermediates(declination=delta, latitude=latitude)

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
    k_secs_d = const.day_seconds
    k_Gsc = const.solar_constant

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

    nu, lambda_ = calculate_heliocentric_longitudes(
        ordinal_date=julian_day,
        n_days=n_days,
        solar_eccentricity=const.solar_eccentricity,
        solar_perihelion=const.solar_perihelion,
    )

    dr = calculate_distance_factor(nu=nu, solar_eccentricity=const.solar_eccentricity)

    delta = calc_declination_angle_delta(
        lambda_=lambda_, k_eps=const.solar_obliquity, k_pir=const.k_pir
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


def calculate_rw_intermediate(
    transmissivity: NDArray[np.float64],
    distance_ratio: NDArray[np.float64],
    shortwave_albedo: float = CoreConst().shortwave_albedo,
    solar_constant: float = CoreConst().solar_constant,
) -> NDArray[np.float64]:
    r"""Calculate the rw intermediate variable.

    This function calculates the widely used variable substitute ``rw`` (:math:`r_w`, W
    m^-2) as:

    .. math::

        r_w = (1.0 - k_{alb_sw}) \tau  k_{Gsc} d_r

    Args:
        transmissivity: Bulk transmissivity, (:math:`\tau`, unitless).
        distance_ratio: Distance ratio, unitless
        shortwave_albedo: The shortwave albedo
        solar_constant: The solar constant, W/m^2

    Returns:
        An array of intermediate variable rw, :math:`W/m^2`.
    """

    return (1.0 - shortwave_albedo) * transmissivity * solar_constant * distance_ratio


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

    ru, rv = calculate_ru_rv_intermediates(declination=delta, latitude=latitude)
    rw = calculate_rw_intermediate(
        transmissivity=tau,
        distance_ratio=dr,
        shortwave_albedo=const.shortwave_albedo,
        solar_constant=const.solar_constant,
    )

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


def calculate_daytime_net_radiation(
    net_longwave_radiation: NDArray[np.float64],
    crossover_hour_angle: NDArray[np.float64],
    sunset_hour_angle: NDArray[np.float64],
    declination: NDArray[np.float64],
    latitude: NDArray[np.float64],
    transmissivity: NDArray[np.float64],
    distance_ratio: NDArray[np.float64],
    shortwave_albedo: float = CoreConst().shortwave_albedo,
    solar_constant: float = CoreConst().solar_constant,
    day_seconds: float = CoreConst().day_seconds,
) -> NDArray:
    r"""Calculate daily net radiation.

    Calculates the daily net radiation , :math:`R_{nd}`, :math:`J/m^2` as:

    .. math::

        R_{d} = \left( \frac{n_s}}{\pi} \right)
            \left( h_n  (r_w \cdot r_u - R_{nl}) + r_w  r_v  sin(h_n)) \right)

    Args:
        net_longwave_radiation: Net longwave radiation, (:math:`R_{nl}`, W m-2)
        crossover_hour_angle: Crossover hour angle, (:math:`h_n`, degrees)
        sunset_hour_angle: sunset hour angle, (:math:`h_s`, degrees)
        declination: solar declination (:math:`\delta`, degrees)
        latitude: Site latitude (:math:`\phi`, degrees)
        transmissivity: Bulk transmissivity (:math:`\tau`, unitless)
        distance_ratio: distance ratio (:math:`d_r`, unitless)
        shortwave_albedo: The shortwave albedo
        solar_constant: The solar constant, W/m^2
        day_seconds: Number of seconds in one solar day (:math:`n_s`, seconds)

    Result:
        _calc_daytime_net_radiation
    """
    ru, rv = calculate_ru_rv_intermediates(declination=declination, latitude=latitude)
    rw = calculate_rw_intermediate(
        transmissivity=transmissivity,
        distance_ratio=distance_ratio,
        shortwave_albedo=shortwave_albedo,
        solar_constant=solar_constant,
    )

    return _calculate_daytime_net_radiation(
        rw=rw,
        rv=ru,
        ru=rv,
        crossover_hour_angle=crossover_hour_angle,
        net_longwave_radiation=net_longwave_radiation,
        day_seconds=day_seconds,
    )


def _calculate_daytime_net_radiation(
    rw: NDArray[np.float64],
    rv: NDArray[np.float64],
    ru: NDArray[np.float64],
    crossover_hour_angle: NDArray[np.float64],
    net_longwave_radiation: NDArray[np.float64],
    day_seconds: float = CoreConst().day_seconds,
) -> NDArray:
    """Calculates daily net radiation, :math:`J/m^2`.

    Args:
        rw: dimensionless variable substitute
        rv: dimensionless variable substitute
        ru: dimensionless variable substitute
        crossover_hour_angle: crossover hour angle, degrees
        net_longwave_radiation: net longwave radiation, W/m^2
        day_seconds: seconds in one solar day

    Result:
        An array of daily net radiation, J/m^2
    """

    rn_d = (day_seconds / np.pi) * (
        np.deg2rad(crossover_hour_angle) * (rw * ru - net_longwave_radiation)
        + rw * rv * np.sin(np.deg2rad(crossover_hour_angle))
    )

    return rn_d


def calculate_nighttime_net_radiation(
    net_longwave_radiation: NDArray[np.float64],
    crossover_hour_angle: NDArray[np.float64],
    sunset_hour_angle: NDArray[np.float64],
    declination: NDArray[np.float64],
    latitude: NDArray[np.float64],
    transmissivity: NDArray[np.float64],
    distance_ratio: NDArray[np.float64],
    shortwave_albedo: float = CoreConst().shortwave_albedo,
    solar_constant: float = CoreConst().solar_constant,
    day_seconds: float = CoreConst().day_seconds,
) -> NDArray[np.float64]:
    r"""Calculates nightime net radiation.

    This function calculates nighttime net radiation (:math:`R_{nn}`, J m-2) as:

    .. math::

        R_{nn} = \left(
            r_w  r_v (\sin(h_s) - \sin(h_n))
            + r_w  r_u  (h_s - h_n)
            - R_{nl} (\pi -  h_n)
            \right)  \left( \frac{n_s}{\pi} \right),

    where :math:`r_u`, :math:`r_v` and :math:`r_w` are the outputs of
    :meth:`calculate_ru_rv_intermediates` and :meth:`calculate_rw_intermediate`.

    Args:
        net_longwave_radiation: Net longwave radiation, (:math:`R_{nl}`, W m-2)
        crossover_hour_angle: Crossover hour angle, (:math:`h_n`, degrees)
        sunset_hour_angle: sunset hour angle, (:math:`h_s`, degrees)
        declination: solar declination (:math:`\delta`, degrees)
        latitude: Site latitude (:math:`\phi`, degrees)
        transmissivity: Bulk transmissivity (:math:`\tau`, unitless)
        distance_ratio: distance ratio (:math:`d_r`, unitless)
        shortwave_albedo: The shortwave albedo
        solar_constant: The solar constant, W/m^2
        day_seconds: Number of seconds in one solar day (:math:`n_s`, seconds)

    Returns:
        An array of nighttime net radiation, :math:`J/m^2`
    """
    ru, rv = calculate_ru_rv_intermediates(declination=declination, latitude=latitude)
    rw = calculate_rw_intermediate(
        transmissivity=transmissivity,
        distance_ratio=distance_ratio,
        shortwave_albedo=shortwave_albedo,
        solar_constant=solar_constant,
    )

    return _calculate_nighttime_net_radiation(
        rw=rw,
        rv=rv,
        ru=ru,
        sunset_hour_angle=sunset_hour_angle,
        crossover_hour_angle=crossover_hour_angle,
        net_longwave_radiation=net_longwave_radiation,
        day_seconds=day_seconds,
    )


def _calculate_nighttime_net_radiation(
    rw: NDArray[np.float64],
    rv: NDArray[np.float64],
    ru: NDArray[np.float64],
    sunset_hour_angle: NDArray[np.float64],
    crossover_hour_angle: NDArray[np.float64],
    net_longwave_radiation: NDArray[np.float64],
    day_seconds: float = CoreConst().day_seconds,
) -> NDArray[np.float64]:
    """Calculates nightime net radiation using precalculated intermediates.

    This function calculates nighttime net radiation (:math:`R_{nn}` :math:`J/m^2`),
    and differs from :math:`calculate_nighttime_net_radiation` in requiring
    precalculated values of :math:`r_u`, :math:`r_v` and :math:`r_w` using
    :meth:`calculate_ru_rv_intermediates` and :meth:`calculate_rw_intermediate`. These
    values are shared across several solar functions and so it is more efficient to be
    able to provide these as precalculated inputs.

    Args:
        rw: intermediate variable (:math:`r_w`, dimensionless)
        rv: intermediate variable (:math:`r_v`, dimensionless)
        ru: intermediate variable (:math:`r_u`, dimensionless)
        sunset_hour_angle: Sunset hour angle (:math:`h_s`, degrees).
        crossover_hour_angle: Crossover hour angle (:math:`h_n`, degrees)
        net_longwave_radiation: Net longwave radiation, (:math:`r_n`, W m-2)
        day_seconds: Seconds in one solar day (:math:`d_s`, seconds)

    Returns:
        An array of nighttime net radiation, :math:`J/m^2`
    """

    return (
        (
            rw
            * rv
            * (
                np.sin(np.deg2rad(sunset_hour_angle))
                - np.sin(np.deg2rad(crossover_hour_angle))
            )
        )
        + (rw * ru * np.deg2rad(sunset_hour_angle - crossover_hour_angle))
        - (net_longwave_radiation * (np.pi - np.deg2rad(crossover_hour_angle)))
    ) * (day_seconds / np.pi)


def calculate_heliocentric_longitudes(
    ordinal_date: NDArray[np.float64],
    n_days: NDArray[np.float64],
    solar_eccentricity: float = CoreConst().solar_eccentricity,
    solar_perihelion: float = CoreConst().solar_perihelion,
) -> tuple[NDArray, NDArray]:
    r"""Calculate heliocentric longitude and anomaly.

    This function calculates the heliocentric true anomaly (``nu``, degrees) and true
    longitude (``lambda_``, degrees), given the Julian day in the year and the
    number of days in the year, following :cite:t:`berger:1978a`.

    Args:
        ordinal_date: The ordinal date
        n_days: The number of days in the year
        solar_eccentricity: The solar eccentricity (:math:`e`), defaulting to
            :attr:`~pyrealm.constants.CoreConst.solar_eccentricity`.
        solar_perihelion: The solar perihelion (:math:`\omega`), defaulting to
            :attr:`~pyrealm.constants.CoreConst.solar_perihelion`.

    Returns:
        A tuple of NDArrays containing ``nu`` and ``lambda_``.
    """

    # TODO - a lot of wildly unnecessary unit changing here. Can we just work in radians
    #        from the constants up?

    # Variable substitutes:
    eccen_sq = solar_eccentricity**2
    eccen_cb = solar_eccentricity**3
    xse = np.sqrt(1.0 - eccen_sq)

    # Mean longitude for vernal equinox in degrees:
    xlam = np.rad2deg(
        (
            (
                (solar_eccentricity / 2.0 + eccen_cb / 8.0)
                * (1.0 + xse)
                * np.sin(np.deg2rad(solar_perihelion))
            )
            - (
                eccen_sq
                / 4.0
                * (0.5 + xse)
                * np.sin(np.deg2rad(2.0 * solar_perihelion))
            )
            + (
                eccen_cb
                / 8.0
                * (1.0 / 3.0 + xse)
                * np.sin(np.deg2rad(3.0 * solar_perihelion))
            )
        )
        * 2.0
    )

    # Mean longitude for day of year in degrees:
    dlamm = xlam + (ordinal_date - 80.0) * (360.0 / n_days)

    # Mean anomaly in radians:
    ranm = np.deg2rad(dlamm - solar_perihelion)

    # True anomaly in radians:
    ranv = (
        ranm
        + ((2.0 * solar_eccentricity - eccen_cb / 4.0) * np.sin(ranm))
        + (5.0 / 4.0 * eccen_sq * np.sin(2.0 * ranm))
        + (13.0 / 12.0 * eccen_cb * np.sin(3.0 * ranm))
    )

    # True longitude in degrees constrained to 0 - 360
    lambda_ = (np.rad2deg(ranv) + solar_perihelion) % 360

    # True anomaly in degrees constrained to 0 - 360
    nu = (lambda_ - solar_perihelion) % 360

    return nu, lambda_


def calc_solar_elevation(site_obs_data: LocationDateTime) -> NDArray:
    r"""Calculate the solar elevation angle for a specific location and times.

    This function calculates the solar elevation angle, which is the angle between the 
    sun and the observer's local horizon, using the methods outlined in 
    :cite:t:`depury:1997a`.

    NB: This implementation does not correct for the effect of local observation 
    altitude on perceived solar elevation.

    This approach uses the following calculations:

        - :func:`calculate_day_angle`
        - :func:`calculate_equation_of_time`
        - :func:`calculate_solar_noon`
        - :func:`calculate_local_hour_angle`
        - :func:`calculate_solar_declination`
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

    G_d = calculate_day_angle(site_obs_data.julian_days)

    E_t = calculate_equation_of_time(G_d)

    t0 = calculate_solar_noon(
        site_obs_data.longitude, site_obs_data.local_standard_meridian, E_t
    )

    hour_angle = calculate_local_hour_angle(site_obs_data.decimal_time, t0)

    declination = calculate_solar_declination(site_obs_data.julian_days)

    elevation = elevation_from_lat_dec_hn(
        site_obs_data.latitude_rad, declination, hour_angle
    )
    return elevation


def calculate_solar_elevation_angle(
    latitude: NDArray[np.float64],
    declination: NDArray[np.float64],
    hour_angle: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate the solar elevation angle.

    The solar elevation angle (:math:`\alpha`) is the angle between the horizon and the
    sun, giving the height of the sun in the sky at a given time. This function
    calculates the elevation angle based on the observer's latitude (:math:`\phi`) , the
    solar declination (:math:`\delta`), and the hour angle (:math:`h`), following Eqn
    A13 of :cite:t:`depury:1997a`:

    .. math::

        \sin(\alpha) = \sin(\phi)  \sin(\delta) + \cos(\phi) \cos(\delta) \cos(h)


    The elevation angle is then given by:

    .. math::
        \alpha = \arcsin(\sin(\alpha))

    Args:
        latitude: Array of latitudes in radians.
        declination: Array of solar declination angles in radians.
        hour_angle: Array of hour angles in radians.

    Returns:
        Solar elevation angles in radians.

    """

    return np.arcsin(
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )


def calculate_solar_declination(
    ordinal_date: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate solar declination angle.

    Calculates the solar declination angle (:math:`\delta`, rad) from the ordinal date,
    following Eqn A14 of :cite:t:`depury:1997a`.

    .. math::

        \delta = -23.4 \frac{\pi}{180} \cos \left(\frac{2 \pi  (td + 10)}{365}\right)

    Args:
        ordinal_date: The ordinal dates of observations.

    Returns:
        Solar declination angles in radians.
    """

    return -23.4 * (np.pi / 180) * np.cos((2 * np.pi * (ordinal_date + 10)) / 365)


def calculate_local_hour_angle(
    current_time: NDArray[np.float64], solar_noon: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Calculate the local hour angle (:math:`h`).

    The local hour angle (:math:`h`) is a measure of time, expressed as an angle in
    radians, that indicates the position of the sun relative to solar noon. This
    function calculates the local hour angle following equation A15 of
    :cite:t:`depury:1997a`.

    .. math::

        h = \pi \frac{t - t_{0}}{12}

    Args:
        current_time: Current time values in decimal hours (:math:`t`).
        solar_noon: Solar noon time values in decimal hours (:math`t_0`).

    Returns:
        The local hour angle in radians

    """

    return np.pi * (current_time - solar_noon) / 12


def calculate_solar_noon(
    longitude: NDArray[np.float64],
    equation_of_time: NDArray[np.float64],
    standard_longitude: float = 0,
) -> NDArray:
    r"""Calculate the solar noon  for a given location.

    The solar noon (:math:`t_0`) is the time of day when the sun is at its highest point
    in the sky for a given location. This function calculates the solar noon by
    adjusting the standard noon time (12:00 PM) given the local longitude and the
    equation of time for the day of observation, following EqA16,
    :cite:t:`depury:1997a`.

    .. math::

        t_{0} = 12 + \frac{4 \cdot -(L_{e} - L_{s}) - E_{t}}{60}

    Args:
        longitude: The local longitude in degrees (:math:`L_e`).
        equation_of_time: The equation of time given the ordinal date of the
            observation (:math:`E_t`).
        standard_longitude: The standard meridian for the observation degrees,
            defaulting to the Greenwich meridan (:math:`L_s`).

    Returns:
        The solar noon time in hours (as a floating-point number), which can be
        interpreted as a time of day.

    """

    return 12 + (4 * -(longitude - standard_longitude) - equation_of_time) / 60


def calculate_equation_of_time(
    day_angle: NDArray[np.float64],
    coef: tuple[float, ...] = CoreConst.equation_of_time_coef,
) -> NDArray[np.float64]:
    r"""Calculate the equation of time.

    Calculates the equation of time in minutes from the day angle (:math:$\Gamma$, rad).

    .. math::

        E_t = f \left(
            a + b \cos(\Gamma) + c \sin(\Gamma) + d \cos(2\Gamma) + e \sin(2\Gamma)
            \right),

    where :math:`a,b,c,d,e,f` are the coefficients of the equation, defaulting to the
    values in :attr:`~pyrealm.constants.CoreConst.equation_of_time_coef` taken from
    Eqn 1.4.1 of :cite:t:`iqbal:1983a`. Note that Eqn A17 of :cite:t:`depury:1997a`
    provides an implementation that contains errors.

    Args:
        day_angle: The day angle in radians
        coef: A tuple of coefficients for the equation of time.

    Returns:
        An array of equation of time values (minutes)
    """

    a, b, c, d, e, f = coef

    return (
        a
        + b * np.cos(day_angle)
        + c * np.sin(day_angle)
        + d * np.cos(2 * day_angle)
        + e * np.sin(2 * day_angle)
    ) * f


def calculate_day_angle(ordinal_date: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate the solar day angle.

    Calculates the solar day angle (``gamma``, :math:`\Gamma`, rad) for ordinal dates
    ('Julian dates') using Eqn A18 of :cite:t:`depury:1997a`.

    .. math::

        \Gamma = \frac{2\pi (N - 1)}{365}

    Args:
        ordinal_date: The ordinal date for which to calculate the day angle.

    Returns:
        An array of solar day angles in radians.
    """

    return 2 * np.pi * (ordinal_date - 1) / 365
