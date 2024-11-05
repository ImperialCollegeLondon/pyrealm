"""The ``solar`` submodule provides functions and classes to calculate daily solar
radiation fluxes and other radiative values.
"""  # noqa: D205

from dataclasses import InitVar, dataclass, field

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.calendar import Calendar
from pyrealm.core.solar import (
    calc_daily_solar_radiation,
    calc_daytime_net_radiation,
    calc_declination_angle_delta,
    calc_distance_factor,
    calc_heliocentric_longitudes,
    calc_lat_delta_intermediates,
    calc_net_longwave_radiation,
    calc_net_rad_crossover_hour_angle,
    calc_nighttime_net_radiation,
    calc_ppfd_from_tau_ra_d,
    calc_rw,
    calc_sunset_hour_angle,
    calc_transmissivity,
)
from pyrealm.core.utilities import check_input_shapes


@dataclass
class DailySolarFluxes:
    """Calculate daily solar fluxes.

    This dataclass takes arrays describing the latitude, elevation, sunshine fraction
    and mean daily temperature for observations and then calculates key radiation fluxes
    given a Calendar object providing the Julian day of the observations and the year
    and number of days in the year.

    Args:
        lat: The Latitude of observations (degrees)
        elv: Elevation of observations, metres
        dates: Dates of observations
        sf: Daily sunshine fraction of observations, unitless
        tc: Daily temperature of observations, °C
    """

    lat: InitVar[NDArray]
    elv: InitVar[NDArray]
    dates: Calendar
    sf: InitVar[NDArray]
    tc: InitVar[NDArray]
    core_const: CoreConst = field(default_factory=lambda: CoreConst())

    nu: NDArray[np.float64] = field(init=False)
    """True heliocentric anomaly, degrees"""
    lambda_: NDArray[np.float64] = field(init=False)
    """True heliocentric longitude, degrees"""
    dr: NDArray[np.float64] = field(init=False)
    """Distance factor, -"""
    delta: NDArray[np.float64] = field(init=False)
    """Declination angle, degrees"""
    ru: NDArray[np.float64] = field(init=False)
    """Intermediate variable, unitless"""
    rv: NDArray[np.float64] = field(init=False)
    """Intermediate variable, unitless"""
    hs: NDArray[np.float64] = field(init=False)
    """Sunset hour angle, degrees"""
    ra_d: NDArray[np.float64] = field(init=False)
    """Daily extraterrestrial solar radiation, J/m^2"""
    tau: NDArray[np.float64] = field(init=False)
    """Transmittivity, unitless"""
    ppfd_d: NDArray[np.float64] = field(init=False)
    """Daily PPFD, mol/m^2"""
    rnl: NDArray[np.float64] = field(init=False)
    """Net longwave radiation, W/m^2"""
    rw: NDArray[np.float64] = field(init=False)
    """Intermediate variable,  W/m^2"""
    hn: NDArray[np.float64] = field(init=False)
    """Net radiation cross-over hour angle, degrees"""
    rn_d: NDArray[np.float64] = field(init=False)
    """Daytime net radiation, J/m^2"""
    rnn_d: NDArray[np.float64] = field(init=False)
    """Nighttime net radiation (rnn_d), J/m^2"""

    def __post_init__(
        self,
        lat: NDArray[np.float64],
        elv: NDArray[np.float64],
        sf: NDArray[np.float64],
        tc: NDArray[np.float64],
    ) -> None:
        """Populates key fluxes from input variables."""

        # Validate the inputs
        shapes = check_input_shapes(lat, elv, sf, tc)
        if len(self.dates) != shapes[0]:
            raise ValueError(
                "The calendar is not the same length as the first axis of inputs "
            )

        # Calculate heliocentric longitudes (nu and lambda), Berger (1978)
        nu, lambda_ = calc_heliocentric_longitudes(
            self.dates.julian_day, self.dates.days_in_year
        )

        # Calculate distance factor (dr), Berger et al. (1993)
        dr = calc_distance_factor(nu, self.core_const.k_e)

        # Calculate declination angle (delta), Woolf (1968)
        delta = calc_declination_angle_delta(
            lambda_, self.core_const.k_eps, self.core_const.k_pir
        )

        # The nu, lambda_, dr and delta attributes are all one dimensional arrays
        # calculated from the Calendar along the first (time) axis of the other inputs.
        # These need to be broadcastable to the shape of the other inputs. The
        # expand_dims variable gets a list of the axes to expand onto - which will be an
        # empty list when ndim=1, leaving the targets unchanged.
        expand_dims = list(np.arange(1, elv.ndim))
        self.nu = np.expand_dims(nu, axis=expand_dims)
        self.lambda_ = np.expand_dims(lambda_, axis=expand_dims)
        self.dr = np.expand_dims(dr, axis=expand_dims)
        self.delta = np.expand_dims(delta, axis=expand_dims)

        self.ru, self.rv = calc_lat_delta_intermediates(self.delta, lat)

        # Calculate the sunset hour angle (hs), Eq. 3.22, Stine & Geyer (2001)
        self.hs = calc_sunset_hour_angle(self.delta, lat, self.core_const.k_pir)

        # Calculate daily extraterrestrial solar radiation (ra_d), J/m^2
        # Eq. 1.10.3, Duffy & Beckman (1993)
        self.ra_d = calc_daily_solar_radiation(
            self.dr, self.hs, self.delta, lat, self.core_const
        )

        # Calculate transmittivity (tau), unitless
        # Eq. 11, Linacre (1968); Eq. 2, Allen (1996)
        self.tau = calc_transmissivity(
            sf, elv, self.core_const.k_c, self.core_const.k_d
        )

        self.rw = calc_rw(
            self.tau, self.dr, self.core_const.k_alb_sw, self.core_const.k_Gsc
        )

        # Calculate daily PPFD (ppfd_d), mol/m^2
        self.ppfd_d = calc_ppfd_from_tau_ra_d(
            self.tau, self.ra_d, self.core_const.k_fFEC, self.core_const.k_alb_vis
        )

        # Estimate net longwave radiation (rnl), W/m^2
        # Eq. 11, Prentice et al. (1993); Eq. 5 and 6, Linacre (1968)
        self.rnl = calc_net_longwave_radiation(
            sf, tc, self.core_const.k_b, self.core_const.k_A
        )

        # Calculate net radiation cross-over hour angle (hn), degrees
        self.hn = calc_net_rad_crossover_hour_angle(
            self.rnl, self.tau, self.dr, self.delta, lat, self.core_const
        )

        # Calculate daytime net radiation (rn_d), J/m^2
        self.rn_d = calc_daytime_net_radiation(
            self.hn, self.rnl, self.delta, lat, self.tau, self.dr, self.core_const
        )

        # Calculate nighttime net radiation (rnn_d), J/m^2
        self.rnn_d = calc_nighttime_net_radiation(
            self.rnl,
            self.hn,
            self.hs,
            self.delta,
            lat,
            self.tau,
            self.dr,
            self.core_const,
        )
