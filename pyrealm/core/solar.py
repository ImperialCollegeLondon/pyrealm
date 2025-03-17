"""The :mod:`~pyrealm.core.solar` submodule provides functions to calculate
photosynthetic photon flux density (PPFD), daily solar radiation fluxes and other
radiative values.

Several of the calculations share a set of intermediate values (:math:`r_u, r_v, r_w`),
which are calculated using :meth:`calculate_ru_rv_intermediates` and
:meth:`calculate_rw_intermediate`. The functions for these calculations have a public
function that uses standard input variables, and a second private version of the
function that accepts precalculated values of :math:`r_u, r_v, r_w`. These private
functions can be used to avoid repeated calculation of intermediate values where
multiple functions need to be run.
"""  # noqa: D205

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
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


def calculate_solar_declination_angle(
    lambda_: NDArray[np.float64],
    solar_obliquity: float = CoreConst().solar_obliquity,
) -> NDArray[np.float64]:
    r"""Calculate the solar declination angle.

    This function calculates the solar declination angle (:math:`\delta`, degrees)
    following :cite:t:`Woolf:1968`.

    .. math::

        \delta = \arcsin(\sin(\lambda)\sin(\epsilon))

    Args:
        lambda_: The heliocentric longitude (:math:`\lambda`, degrees)
        solar_obliquity: Solar obliquity (:math:`\epsilon`)

    Returns:
        An array of solar declination angle delta
    """

    return np.rad2deg(
        np.arcsin(np.sin(np.deg2rad(lambda_)) * np.sin(np.deg2rad(solar_obliquity)))
    )


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


def calculate_sunset_hour_angle(
    declination: NDArray[np.float64], latitude: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Calculate the sunset hour angle.

    This function calculates the sunset hour angle :math:`h_s` in degrees following Eq.
    3.22 of :cite:t:`stine_geyer:2001` as:

    .. math::

        hs = \arccos(-1.0  \frac{ru}{rv}),

    where :math:`r_u`and :math:`r_v` are the outputs of
    :meth:`calculate_ru_rv_intermediates`.


    The function is clamped within :math:`[-180°, 180°]`.

    Args:
        declination: The solar declination angle (:math:`\delta`, degrees)
        latitude: The site latitude (:math:`\phi`, degrees)

    Returns:
        An array of local hour angle, degrees
    """

    ru, rv = calculate_ru_rv_intermediates(declination=declination, latitude=latitude)

    return _calculate_sunset_hour_angle(ru, rv)


def _calculate_sunset_hour_angle(
    ru: NDArray[np.float64],
    rv: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate sunset hour angle from intermediates.

    This function calculates the sunset hour angle using Eq3.22,
    :cite:t:'stine_geyer:2001', using precalculated intermediate values (see
    :meth:`calculate_ru_rv_intermediates`)

    Args:
        ru: dimensionless parameter
        rv: dimensionless parameter

    Returns:
        An array of local hour angle, degrees
    """

    return np.rad2deg(np.arccos(-1.0 * np.clip(ru / rv, -1.0, 1.0)))


def calculate_daily_solar_radiation(
    distance_factor: NDArray[np.float64],
    sunset_hour_angle: NDArray[np.float64],
    declination: NDArray[np.float64],
    latitude: NDArray[np.float64],
    day_seconds: float = CoreConst().day_seconds,
    solar_constant: float = CoreConst().solar_constant,
) -> NDArray[np.float64]:
    r"""Calculate daily extraterrestrial solar radiation (J/m^2).

    This function calculates the daily extraterrestrial solar radition (:math:`R_d`, J
    m-2) using Eq. 1.10.3 of :cite:t:`Duffie:2013a`.

    .. math::

        R_d = \left( \frac{n_s}{\pi} \right)
            G_{sc} d_r \left(r_u h_s + r_v \sin(h_s) \right)

    Args:
        distance_factor: The distance factor (:math:`d_r`, unitless)
        sunset_hour_angle: Local hour angle, (:math:`h_s`, degrees)
        declination: solar declination (:math:`\delta`, degrees)
        latitude: Site latitude (:math:`\phi`, degrees)
        day_seconds: Number of seconds in one solar day (:math:`n_s`, seconds)
        solar_constant: The solar constant (:math:`G_{sc}`, W m-2)

    Returns:
        An array of daily solar radiation, J/m^2
    """

    ru, rv = calculate_ru_rv_intermediates(declination=declination, latitude=latitude)

    return _calculate_daily_solar_radiation(
        ru=ru,
        rv=rv,
        distance_factor=distance_factor,
        sunset_hour_angle=sunset_hour_angle,
        day_seconds=day_seconds,
        solar_constant=solar_constant,
    )


def _calculate_daily_solar_radiation(
    ru: NDArray[np.float64],
    rv: NDArray[np.float64],
    distance_factor: NDArray[np.float64],
    sunset_hour_angle: NDArray[np.float64],
    day_seconds: float = CoreConst().day_seconds,
    solar_constant: float = CoreConst().solar_constant,
) -> NDArray[np.float64]:
    """Calculate daily extraterrestrial solar radiation from intermediate values.

    This function calculates the daily extraterrestrial solar radition (:math:R_d`, J
    m-2) following Eq. 1.10.3 of :cite:t:`Duffie:2013a` and using precalculated
    intermediate values (see :meth:`calculate_ru_rv_intermediates`)

    Args:
        ru: dimensionless variable substitute
        rv: dimensionless variable substitute
        distance_factor: The distance factor (:math:`d_r`, unitless)
        sunset_hour_angle: Local hour angle, (:math:`h_s`, degrees)
        day_seconds: Number of seconds in one solar day (:math:`n_s`, seconds)
        solar_constant: The solar constant, W/m^2

    Returns:
        An array of daily solar radiation, J/m^2
    """

    return (
        (day_seconds / np.pi)
        * solar_constant
        * distance_factor
        * (
            ru * np.deg2rad(sunset_hour_angle)
            + rv * np.sin(np.deg2rad(sunset_hour_angle))
        )
    )


def calculate_transmissivity(
    sunshine_fraction: NDArray[np.float64],
    elevation: NDArray[np.float64],
    coef: tuple[float, ...] = CoreConst.transmissivity_coef,
) -> NDArray[np.float64]:
    r"""Calculate atmospheric transmissivity.

    This function calculates atmospheric transmissivity (:math:`\tau`, unitless)
    following Eqn. 11 of :cite:t:`Linacre:1968a` and Eq 2 of :cite:t:`allen:1996a`.

    .. math::

        \tau = (c + d s_f) (1.0 + f H)

    Args:
        sunshine_fraction: Daily sunshine fraction of observations, (:math:`s_f`,
            unitless)
        elevation: Elevation of observations, (:math:`H`, metres)
        coef: Coefficients of the equation

    Returns:
        An array of bulk transmissivity, unitless
    """
    c, d, f = coef
    return (c + d * sunshine_fraction) * (1.0 + f * elevation)


def calculate_ppfd_from_tau_rd(
    transmissivity: NDArray[np.float64],
    daily_solar_radiation: NDArray[np.float64],
    visible_light_albedo: float = CoreConst().visible_light_albedo,
    swdown_to_ppfd_factor: float = CoreConst().swdown_to_ppfd_factor,
) -> NDArray[np.float64]:
    r"""Calculate photosynthetic photon flux density from intermediates.

    This function calculates photosynthetic photon flux density (PPFD, µmol m-2 s-1)
    from precalculated values for daily solar radiation and transimissivity.

    .. math::

        \mathrm{PPFD} = (1.0 \times 10^{-6})  f_{PPFD} (1.0 - A_{vis}) \tau  R_{d}

    Args:
        transmissivity: The bulk transmissivity (:math:`\tau`, unitless)
        daily_solar_radiation: Daily solar radiation (:math:`R_d`, J m-2)
        visible_light_albedo: The visible light albedo (:math:`A_{vis}`, unitless)
        swdown_to_ppfd_factor: Conversion factor from W m-2 of sunlight to µmol m-2 s-1
            of PPFD (:math:`f_{PPFD}`).

    Returns:
        An array of photosynthetic photon flux density, µmol m-2 s-1.
    """

    ppfd = (
        (1.0e-6)
        * swdown_to_ppfd_factor
        * (1.0 - visible_light_albedo)
        * transmissivity
        * daily_solar_radiation
    )

    return ppfd


def calculate_ppfd(
    sunshine_fraction: NDArray[np.float64],
    elevation: NDArray[np.float64],
    latitude: NDArray[np.float64],
    ordinal_date: NDArray[np.int_],
    n_days: NDArray[np.int_],
    const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    r"""Calculates photosynthetic photon flux density (PPFD).

    This function calculates photosynthetic photon flux density (PPFD, mol m-2) for a
    given ordinal date in a year for a location at a given latitude and elevation. The
    PPFD value is moderated by the sunshine fraction for the day to account for effects
    of cloud cover.

    Args:
        sunshine_fraction: Daily sunshine fraction of observations, unitless.
        elevation: Elevation of observations, metres.
        latitude: The latitude of observations (degrees).
        ordinal_date: The ordinal_date of the observation.
        n_days: Days in the calendar year.
        const: CoreConst object.

    Returns:
        An array of photosynthetic photon flux density, :math:`PPFD` (:math:`mol/m^2`).

    Example:
        >>> # Define variable values
        >>> sf = np.array([1.0])
        >>> elv = np.array([142])
        >>> latitude = np.array([37.7])
        >>> julian_day = np.array([172])
        >>> n_days = np.array([366])
        >>> # Evaluate function
        >>> calc_ppfd(
        ...    sf=sf, elv=elv, latitude=latitude, julian_day=julian_day, n_days=n_days
        ... )
        array([62.04230021])
    """

    # Validate the inputs
    _ = check_input_shapes(latitude, elevation, sunshine_fraction, ordinal_date, n_days)

    # Calculate intermediate values

    nu, lambda_ = calculate_heliocentric_longitudes(
        ordinal_date=ordinal_date,
        n_days=n_days,
        solar_eccentricity=const.solar_eccentricity,
        solar_perihelion=const.solar_perihelion,
    )

    distance_factor = calculate_distance_factor(
        nu=nu, solar_eccentricity=const.solar_eccentricity
    )

    declination = calculate_solar_declination_angle(
        lambda_=lambda_, solar_obliquity=const.solar_obliquity
    )

    tau = calculate_transmissivity(
        sunshine_fraction=sunshine_fraction,
        elevation=elevation,
        coef=const.transmissivity_coef,
    )

    sunset_hour_angle = calculate_sunset_hour_angle(
        declination=declination, latitude=latitude
    )

    ra_d = calculate_daily_solar_radiation(
        distance_factor=distance_factor,
        sunset_hour_angle=sunset_hour_angle,
        declination=declination,
        latitude=latitude,
        day_seconds=const.day_seconds,
        solar_constant=const.solar_constant,
    )

    # Calculate ppfd

    return calculate_ppfd_from_tau_rd(
        transmissivity=tau,
        daily_solar_radiation=ra_d,
        visible_light_albedo=const.visible_light_albedo,
        swdown_to_ppfd_factor=const.swdown_to_ppfd_factor,
    )


def calculate_net_longwave_radiation(
    sunshine_fraction: NDArray[np.float64],
    temperature: NDArray[np.float64],
    coef: tuple[float, float] = CoreConst().net_longwave_radiation_coef,
) -> NDArray[np.float64]:
    r"""Calculate net longwave radiation.

    This function calculates the net longwave radiation (:math:`R_{nl}`, W m-2)
    following  Eqn. 11 and Table 1 of :cite:t:`colinprentice:1993a`:

    .. math::

        R_{nl} = (b + (1.0 - b) n_i) (A - t_c)

    Args:
        sunshine_fraction: Sunshine fraction of observations (:math:`n_i`, unitless)
        temperature: Temperature of observations (:math:`t_c`, °C).
        coef: Coefficients :math:`b` and :math:`A` of the equation, defaulting to values
            in
            :attr:`~pyrealm.constants.core_const.CoreConst.net_longwave_radiation_coef`

    Returns:
        An array of net longwave radiation.
    """
    b, A = coef
    return (b + (1.0 - b) * sunshine_fraction) * (A - temperature)


def calculate_rw_intermediate(
    transmissivity: NDArray[np.float64],
    distance_factor: NDArray[np.float64],
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
        distance_factor: Distance ratio, unitless
        shortwave_albedo: The shortwave albedo
        solar_constant: The solar constant, W/m^2

    Returns:
        An array of intermediate variable rw, :math:`W/m^2`.
    """

    return (1.0 - shortwave_albedo) * transmissivity * solar_constant * distance_factor


def calculate_net_radiation_crossover_hour_angle(
    net_longwave_radiation: NDArray[np.float64],
    transmissivity: NDArray[np.float64],
    distance_factor: NDArray[np.float64],
    declination: NDArray[np.float64],
    latitude: NDArray[np.float64],
    shortwave_albedo: float = CoreConst().shortwave_albedo,
    solar_constant: float = CoreConst().solar_constant,
) -> NDArray[np.float64]:
    r"""Calculates the net radiation crossover hour angle.

    This function calculates the net radiation crossover hour angle (:math:`h_n`,
    degrees) as:

    .. math::

        h_n = \arccos(\frac{R_{nl} - r_w  r_u}{r_w  r_v}),

    where :math:`r_u`, :math:`r_v` and :math:`r_w` are the outputs of
    :meth:`calculate_ru_rv_intermediates` and :meth:`calculate_rw_intermediate`.

    The function is clamped within :math:`[-180°, 180°]`.

    Args:
        net_longwave_radiation: Net longwave radiation, (:math:`R_{nl}`, W m-2)
        transmissivity: Bulk transmissivity (:math:`\tau`, unitless)
        declination: solar declination (:math:`\delta`, degrees)
        latitude: Site latitude (:math:`\phi`, degrees)
        distance_factor: The distance factor (:math:`d_r`, unitless)
        shortwave_albedo: The shortwave albedo
        solar_constant: The solar constant, W/m^2

    Returns:
        _calc_net_rad_crossover_hour_angle
    """

    ru, rv = calculate_ru_rv_intermediates(declination=declination, latitude=latitude)
    rw = calculate_rw_intermediate(
        transmissivity=transmissivity,
        distance_factor=distance_factor,
        shortwave_albedo=shortwave_albedo,
        solar_constant=solar_constant,
    )

    return _calculate_net_radiation_crossover_hour_angle(
        net_longwave_radiation=net_longwave_radiation,
        rw=rw,
        ru=ru,
        rv=rv,
    )


def _calculate_net_radiation_crossover_hour_angle(
    net_longwave_radiation: NDArray[np.float64],
    rw: NDArray[np.float64],
    ru: NDArray[np.float64],
    rv: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate net radiation crossover hour angle using intermediate values.

    This function calculates the net radiation crossover hour angle (:math:`h_n`
    degrees) using precalculated intermediate values (see
    :meth:`calculate_ru_rv_intermediates` and :meth:`calculate_rw_intermediate`)

    Args:
        rw: dimensionless variable substitute
        rv: dimensionless variable substitute
        ru: dimensionless variable substitute
        net_longwave_radiation: net longwave radiation, W/m^2

    Returns:
        An array of net radiation crossover hour angle, degrees
    """

    return np.rad2deg(
        np.arccos(np.clip((net_longwave_radiation - rw * ru) / (rw * rv), -1.0, 1.0))
    )


def calculate_daytime_net_radiation(
    net_longwave_radiation: NDArray[np.float64],
    crossover_hour_angle: NDArray[np.float64],
    declination: NDArray[np.float64],
    latitude: NDArray[np.float64],
    transmissivity: NDArray[np.float64],
    distance_factor: NDArray[np.float64],
    shortwave_albedo: float = CoreConst().shortwave_albedo,
    solar_constant: float = CoreConst().solar_constant,
    day_seconds: float = CoreConst().day_seconds,
) -> NDArray[np.float64]:
    r"""Calculate daily net radiation.

    Calculates the daily net radiation , :math:`R_{nd}`, :math:`J/m^2` as:

    .. math::

        R_{nd} = \left( \frac{n_s}{\pi} \right)
            \left( h_n  (r_w \cdot r_u - R_{nl}) + r_w r_v sin(h_n)) \right)

    Args:
        net_longwave_radiation: Net longwave radiation, (:math:`R_{nl}`, W m-2)
        crossover_hour_angle: Crossover hour angle, (:math:`h_n`, degrees)
        declination: solar declination (:math:`\delta`, degrees)
        latitude: Site latitude (:math:`\phi`, degrees)
        transmissivity: Bulk transmissivity (:math:`\tau`, unitless)
        distance_factor: The distance factor (:math:`d_r`, unitless)
        shortwave_albedo: The shortwave albedo
        solar_constant: The solar constant, W/m^2
        day_seconds: Number of seconds in one solar day (:math:`n_s`, seconds)

    Result:
        _calc_daytime_net_radiation
    """
    ru, rv = calculate_ru_rv_intermediates(declination=declination, latitude=latitude)
    rw = calculate_rw_intermediate(
        transmissivity=transmissivity,
        distance_factor=distance_factor,
        shortwave_albedo=shortwave_albedo,
        solar_constant=solar_constant,
    )

    return _calculate_daytime_net_radiation(
        rw=rw,
        rv=rv,
        ru=ru,
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
) -> NDArray[np.float64]:
    """Calculates daily net radiation using precalculated intermediates.

    This function calculates daytime net radiation (:math:`R_{nd}` :math:`J/m^2`), using
    precalculated intermediate values (see :meth:`calculate_ru_rv_intermediates` and
    :meth:`calculate_rw_intermediate`)

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
    distance_factor: NDArray[np.float64],
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
        distance_factor: The distance factor (:math:`d_r`, unitless)
        shortwave_albedo: The shortwave albedo
        solar_constant: The solar constant, W/m^2
        day_seconds: Number of seconds in one solar day (:math:`n_s`, seconds)

    Returns:
        An array of nighttime net radiation, :math:`J/m^2`
    """
    ru, rv = calculate_ru_rv_intermediates(declination=declination, latitude=latitude)
    rw = calculate_rw_intermediate(
        transmissivity=transmissivity,
        distance_factor=distance_factor,
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

    This function calculates nighttime net radiation (:math:`R_{nn}` :math:`J/m^2`), a
    using precalculated intermediate values (see :meth:`calculate_ru_rv_intermediates`
    and :meth:`calculate_rw_intermediate`)

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
    ordinal_date: NDArray[np.int_],
    n_days: NDArray[np.int_],
    solar_eccentricity: float = CoreConst().solar_eccentricity,
    solar_perihelion: float = CoreConst().solar_perihelion,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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


# TODO - @davidorme commented this out - issues with typing of usage, which is aimed at
#        single sites. Not currently being used in other functions so parking it while
#        working through revision of the rest of the module and splash.

# def calculate_solar_elevation(
#     site_obs_data: LocationDateTime, const: CoreConst = CoreConst()
# ) -> NDArray[np.float64]:
#     r"""Calculate the solar elevation angle for a specific location and times.

#     This function calculates the solar elevation angle, which is the angle between the
#     sun and the observer's local horizon, using the methods outlined in
#     :cite:t:`depury:1997a`.

#     .. note::

#         This implementation does not correct for the effect of local observation
#         altitude on perceived solar elevation.

#     Args:
#         site_obs_data: A :class:`~pyrealm.core.calendar.LocationDateTime` instance
#             containing the location and time-specific information for the observation
#             site.
#         const: The core constants to be used in calculations.

#     Returns:
#         An array of solar elevation angles in radians, representing the angular height
#         of the sun above the horizon at the specified location and time.

#     Example:
#         >>> # Calculate solar elevation at Wagga Wagga, Australia.
#         >>> # Worked example taken from dePury and Farquhar (1997)
#         >>> from pyrealm.core.calendar import LocationDateTime
#         >>> from pyrealm.core.solar import calculate_solar_elevation
#         >>> # Create instance of LocationDateTime dataclass
#         >>> ldt = LocationDateTime(
#         ...     latitude=-35.058333,
#         ...     longitude=147.34167,
#         ...     year_date_time=np.array([np.datetime64("1995-10-25T10:30")]),
#         ... )
#         >>> # Run solar elevation calculation
#         >>> calculate_solar_elevation(ldt)
#         array([1.0615713])

#     """

#     G_d = calculate_day_angle(ordinal_date=site_obs_data.julian_days)

#     E_t = calculate_equation_of_time(day_angle=G_d, coef=const.equation_of_time_coef)

#     t0 = calculate_solar_noon(
#         longitude=site_obs_data.longitude,
#         equation_of_time=E_t,
#         standard_longitude=site_obs_data.local_standard_meridian,
#     )

#     hour_angle = calculate_local_hour_angle(
#         current_time=site_obs_data.decimal_time, solar_noon=t0
#     )

#     declination = calculate_solar_declination(ordinal_date=site_obs_data.julian_days)

#     elevation = calculate_solar_elevation_angle(
#         latitude=site_obs_data.latitude_rad,
#         declination=declination,
#         hour_angle=hour_angle,
#     )
#     return elevation


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
    ordinal_date: NDArray[np.int_],
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
) -> NDArray[np.float64]:
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


def calculate_day_angle(ordinal_date: NDArray[np.int_]) -> NDArray[np.float64]:
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
