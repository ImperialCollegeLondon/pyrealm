"""The ``solar`` submodule provides the DailySolarFluxes class, used to calculate daily
solar radiation fluxes for observations.
"""  # noqa: D205

from dataclasses import InitVar, dataclass, field

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.calendar import Calendar
from pyrealm.core.solar import (
    _calculate_daily_solar_radiation,
    _calculate_daytime_net_radiation,
    _calculate_net_radiation_crossover_hour_angle,
    _calculate_nighttime_net_radiation,
    _calculate_sunset_hour_angle,
    calculate_distance_factor,
    calculate_heliocentric_longitudes,
    calculate_net_longwave_radiation,
    calculate_ppfd_from_tau_rd,
    calculate_ru_rv_intermediates,
    calculate_rw_intermediate,
    calculate_solar_declination_angle,
    calculate_transmissivity,
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
        latitude: The Latitude of observations (degrees)
        elevation: Elevation of observations (metres)
        dates: Dates of observations
        sunshine_fraction: Daily sunshine fraction of observations (unitless)
        temperature: Daily temperature of observations (°C)
    """

    latitude: InitVar[NDArray[np.float64]]
    elevation: InitVar[NDArray[np.float64]]
    dates: Calendar
    sunshine_fraction: InitVar[NDArray[np.float64]]
    temperature: InitVar[NDArray[np.float64]]
    core_const: CoreConst = field(default_factory=lambda: CoreConst())

    nu: NDArray[np.float64] = field(init=False)
    r"""True heliocentric anomaly (:math:`\nu`, degrees)"""
    lambda_: NDArray[np.float64] = field(init=False)
    r"""True heliocentric longitude, (:math:`\lambda`, degrees)"""
    distance_factor: NDArray[np.float64] = field(init=False)
    """Distance factor (:math:`d_r`, -)"""
    declination: NDArray[np.float64] = field(init=False)
    r"""Declination angle (:math:`\delta`, degrees)"""
    ru: NDArray[np.float64] = field(init=False)
    """Intermediate variable (:math:`r_u`, unitless)"""
    rv: NDArray[np.float64] = field(init=False)
    """Intermediate variable (:math:`r_v`, unitless)"""
    sunset_hour_angle: NDArray[np.float64] = field(init=False)
    """Sunset hour angle (:math:`h_s`, degrees)"""
    daily_solar_radiation: NDArray[np.float64] = field(init=False)
    """Daily extraterrestrial solar radiation (:math:`R_d`, J m-2)"""
    transmissivity: NDArray[np.float64] = field(init=False)
    r"""Transmittivity (:math:`\tau`, unitless)"""
    daily_ppfd: NDArray[np.float64] = field(init=False)
    """Daily photosynthetic photon flux density (PPFD, µmol m-2 s-1)"""
    net_longwave_radiation: NDArray[np.float64] = field(init=False)
    """Net longwave radiation (:math:`R_{nl}`, W m-2)"""
    rw: NDArray[np.float64] = field(init=False)
    """Intermediate variable (:math:`r_w`,  W m-2)"""
    crossover_hour_angle: NDArray[np.float64] = field(init=False)
    """Net radiation cross-over hour angle, (:math:`h_n`, degrees)"""
    daytime_net_radiation: NDArray[np.float64] = field(init=False)
    """Daytime net radiation (:math:`R_{d}`, J m-2)"""
    nighttime_net_radiation: NDArray[np.float64] = field(init=False)
    """Nighttime net radiation (:math:`R_{nn}`, J m-2)"""

    def __post_init__(
        self,
        latitude: NDArray[np.float64],
        elevation: NDArray[np.float64],
        sunshine_fraction: NDArray[np.float64],
        temperature: NDArray[np.float64],
    ) -> None:
        """Populates key fluxes from input variables."""

        # Validate the inputs
        self.shape: tuple = check_input_shapes(
            latitude, elevation, sunshine_fraction, temperature
        )
        """The array shape of the input variables"""

        if self.shape[0] == 1:
            self.shape = (len(self.dates), *self.shape[1:])
        elif self.shape[0] != len(self.dates):
            raise ValueError(
                "The first axis of inputs is neither the same as the calendar or length"
                " one (constant in time)"
            )

        # Calculate heliocentric longitudes (nu and lambda), Berger (1978)
        nu, lambda_ = calculate_heliocentric_longitudes(
            ordinal_date=self.dates.julian_day, n_days=self.dates.days_in_year
        )

        # Calculate distance factor (dr), Berger et al. (1993)
        distance_factor = calculate_distance_factor(
            nu=nu, solar_eccentricity=self.core_const.solar_eccentricity
        )

        # Calculate declination angle (delta), Woolf (1968)
        delta = calculate_solar_declination_angle(
            lambda_=lambda_, solar_obliquity=self.core_const.solar_obliquity
        )

        # The nu, lambda_, distance_factor and declination attributes are all one
        # dimensional arrays calculated from the Calendar along the first (time) axis of
        # the other inputs. These need to be broadcastable to the shape of the other
        # inputs. The expand_dims variable gets a list of the axes to expand onto -
        # which will be an empty list when ndim=1, leaving the targets unchanged.
        expand_dims = list(np.arange(1, len(self.shape)))
        self.nu = np.expand_dims(nu, axis=expand_dims)
        self.lambda_ = np.expand_dims(lambda_, axis=expand_dims)
        self.distance_factor = np.expand_dims(distance_factor, axis=expand_dims)
        self.declination = np.expand_dims(delta, axis=expand_dims)

        # Calculate transmittivity (tau), unitless
        # Eq. 11, Linacre (1968); Eq. 2, Allen (1996)
        self.transmissivity = calculate_transmissivity(
            sunshine_fraction=sunshine_fraction,
            elevation=elevation,
            coef=self.core_const.transmissivity_coef,
        )

        # Calculate intermediate values ru, rv, rw
        self.ru, self.rv = calculate_ru_rv_intermediates(
            declination=self.declination, latitude=latitude
        )

        self.rw = calculate_rw_intermediate(
            transmissivity=self.transmissivity,
            distance_factor=self.distance_factor,
            shortwave_albedo=self.core_const.shortwave_albedo,
            solar_constant=self.core_const.solar_constant,
        )

        # Calculate the sunset hour angle (hs), Eq. 3.22, Stine & Geyer (2001)
        self.sunset_hour_angle = _calculate_sunset_hour_angle(ru=self.ru, rv=self.rv)

        # Calculate daily extraterrestrial solar radiation (R_d, J/m^2)
        # Eq. 1.10.3, Duffy & Beckman (1993)
        self.daily_solar_radiation = _calculate_daily_solar_radiation(
            ru=self.ru,
            rv=self.rv,
            distance_factor=self.distance_factor,
            sunset_hour_angle=self.sunset_hour_angle,
            day_seconds=self.core_const.day_seconds,
            solar_constant=self.core_const.solar_constant,
        )

        # Calculate daily PPFD (ppfd_d), mol/m^2
        self.daily_ppfd = calculate_ppfd_from_tau_rd(
            transmissivity=self.transmissivity,
            daily_solar_radiation=self.daily_solar_radiation,
            swdown_to_ppfd_factor=self.core_const.swdown_to_ppfd_factor,
            visible_light_albedo=self.core_const.visible_light_albedo,
        )

        # Estimate net longwave radiation (rnl), W/m^2
        # Eq. 11, Prentice et al. (1993); Eq. 5 and 6, Linacre (1968)
        self.net_longwave_radiation = calculate_net_longwave_radiation(
            sunshine_fraction=sunshine_fraction,
            temperature=temperature,
            coef=self.core_const.net_longwave_radiation_coef,
        )

        # Calculate net radiation cross-over hour angle (hn), degrees
        self.crossover_hour_angle = _calculate_net_radiation_crossover_hour_angle(
            ru=self.ru,
            rv=self.rv,
            rw=self.rw,
            net_longwave_radiation=self.net_longwave_radiation,
        )

        # Calculate daytime net radiation (rn_d), J/m^2
        self.daytime_net_radiation = _calculate_daytime_net_radiation(
            ru=self.ru,
            rv=self.rv,
            rw=self.rw,
            crossover_hour_angle=self.crossover_hour_angle,
            net_longwave_radiation=self.net_longwave_radiation,
            day_seconds=self.core_const.day_seconds,
        )

        # Calculate nighttime net radiation (rnn_d), J/m^2
        self.nighttime_net_radiation = _calculate_nighttime_net_radiation(
            ru=self.ru,
            rv=self.rv,
            rw=self.rw,
            net_longwave_radiation=self.net_longwave_radiation,
            crossover_hour_angle=self.crossover_hour_angle,
            sunset_hour_angle=self.sunset_hour_angle,
            day_seconds=self.core_const.day_seconds,
        )
