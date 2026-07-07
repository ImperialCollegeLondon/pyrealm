"""The ``solar`` submodule provides the DailySolarFluxes class, used to calculate daily
solar radiation fluxes for observations.
"""  # noqa: D205

from abc import ABC, abstractmethod
from collections.abc import Hashable

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
    calculate_rw_intermediate_from_sw,
    calculate_solar_declination_angle,
    calculate_sunshine_fraction,
    calculate_transmissivity,
)
from pyrealm.core.utilities import check_input_shapes
from pyrealm.core.xarray import ArrayType, get_common_dims, xarray_inputs


class DailySolarFluxesABC(ABC):
    """Calculate daily solar fluxes.

    This dataclass takes arrays describing the latitude, elevation, sunshine fraction
    and mean daily temperature for observations and then calculates key radiation fluxes
    given a Calendar object providing the Julian day of the observations and the year
    and number of days in the year.

    The first dimension for the array inputs should correspond to time. If xarray inputs
    are used, ``dates`` should also be initialised using xarray inputs to
    ensure this. Alternatively, ``latitude`` can include time as the first dimension.

    Args:
        latitude: The Latitude of observations (degrees)
        elevation: Elevation of observations (metres)
        dates: Dates of observations
        sunshine_fraction: Daily sunshine fraction of observations (unitless)
        temperature: Daily temperature of observations (°C)
    """

    @abstractmethod
    def __init__(
        self,
        latitude: ArrayType[np.floating],
        elevation: ArrayType[np.floating],
        temperature: ArrayType[np.floating],
        dates: Calendar,
        core_const: CoreConst = CoreConst(),
        **kwargs: ArrayType[np.floating],
    ):
        self.nu: NDArray[np.floating]
        r"""True heliocentric anomaly (:math:`\nu`, degrees)"""
        self.lambda_: NDArray[np.floating]
        r"""True heliocentric longitude, (:math:`\lambda`, degrees)"""
        self.distance_factor: NDArray[np.floating]
        """Distance factor (:math:`d_r`, -)"""
        self.declination: NDArray[np.floating]
        r"""Declination angle (:math:`\delta`, degrees)"""
        self.ru: NDArray[np.floating]
        """Intermediate variable (:math:`r_u`, unitless)"""
        self.rv: NDArray[np.floating]
        """Intermediate variable (:math:`r_v`, unitless)"""
        self.sunset_hour_angle: NDArray[np.floating]
        """Sunset hour angle (:math:`h_s`, degrees)"""
        self.daily_solar_radiation: NDArray[np.floating]
        """Daily extraterrestrial solar radiation (:math:`R_d`, J m-2)"""
        self.sunshine_fraction: NDArray[np.floating]
        r"""Sunshine fraction (:math:`\tau`, unitless)"""
        self.transmissivity: NDArray[np.floating]
        r"""Transmissivity (:math:`\tau`, unitless)"""
        self.daily_ppfd: NDArray[np.floating]
        """Daily photosynthetic photon flux density (PPFD, µmol m-2 s-1)"""
        self.net_longwave_radiation: NDArray[np.floating]
        """Net longwave radiation (:math:`R_{nl}`, W m-2)"""
        self.rw: NDArray[np.floating]
        """Intermediate variable (:math:`r_w`,  W m-2)"""
        self.crossover_hour_angle: NDArray[np.floating]
        """Net radiation cross-over hour angle, (:math:`h_n`, degrees)"""
        self.daytime_net_radiation: NDArray[np.floating]
        """Daytime net radiation (:math:`R_{d}`, J m-2)"""
        self.nighttime_net_radiation: NDArray[np.floating]
        """Nighttime net radiation (:math:`R_{nn}`, J m-2)"""
        self.dims: list[Hashable]
        """Names of dimensions in any xarray inputs"""
        self.shape: tuple[int, ...]
        """Shape of array inputs"""

        self.dates: Calendar = dates
        self.core_const: CoreConst = core_const

        # Convert any xr.DataArrays to numpy arrays
        self.dims = get_common_dims(latitude, elevation, temperature, *kwargs.values())
        (latitude, elevation, temperature), kw_arrays = xarray_inputs(
            latitude, elevation, temperature, kwargs=kwargs, dims=self.dims
        )

        # Validate the inputs
        self.shape = check_input_shapes(
            latitude, elevation, temperature, *kw_arrays.values()
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

        # Calculate intermediate values ru, rv, rw
        self.ru, self.rv = calculate_ru_rv_intermediates(
            declination=self.declination, latitude=latitude
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


class DailySolarFluxesDavis(DailySolarFluxesABC):
    """Calculate daily solar fluxes.

    This dataclass takes arrays describing the latitude, elevation, sunshine fraction
    and mean daily temperature for observations and then calculates key radiation fluxes
    given a Calendar object providing the Julian day of the observations and the year
    and number of days in the year.

    The first dimension for the array inputs should correspond to time. If xarray inputs
    are used, ``dates`` should also be initialised using xarray inputs to
    ensure this. Alternatively, ``latitude`` can include time as the first dimension.

    Args:
        latitude: The Latitude of observations (degrees)
        elevation: Elevation of observations (metres)
        dates: Dates of observations
        sunshine_fraction: Daily sunshine fraction of observations (unitless)
        temperature: Daily temperature of observations (°C)
    """

    def __init__(
        self,
        latitude: ArrayType[np.floating],
        elevation: ArrayType[np.floating],
        temperature: ArrayType[np.floating],
        sunshine_fraction: ArrayType[np.floating],
        dates: Calendar,
        core_const: CoreConst = CoreConst(longwave_radiation_option="Prentice1993"),
    ) -> None:
        """Populates key fluxes from input variables."""

        super().__init__(
            latitude=latitude,
            elevation=elevation,
            temperature=temperature,
            dates=dates,
            core_const=core_const,
            kwargs=sunshine_fraction,
        )

        self.sunshine_fraction = sunshine_fraction

        # Calculate transmittivity (tau), unitless
        # Eq. 11, Linacre (1968); Eq. 2, Allen (1996)
        self.transmissivity = calculate_transmissivity(
            sunshine_fraction=sunshine_fraction,
            elevation=elevation,
            coef=self.core_const.transmissivity_coef,
        )

        self.rw = calculate_rw_intermediate(
            transmissivity=self.transmissivity,
            distance_factor=self.distance_factor,
            shortwave_albedo=self.core_const.shortwave_albedo,
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


class DailySolarFluxesSandoval(DailySolarFluxesABC):
    """Calculate daily solar fluxes.

    This dataclass takes arrays describing the latitude, elevation, sunshine fraction
    and mean daily temperature for observations and then calculates key radiation fluxes
    given a Calendar object providing the Julian day of the observations and the year
    and number of days in the year.

    The first dimension for the array inputs should correspond to time. If xarray inputs
    are used, ``dates`` should also be initialised using xarray inputs to
    ensure this. Alternatively, ``latitude`` can include time as the first dimension.

    Args:
        latitude: The Latitude of observations (degrees)
        elevation: Elevation of observations (metres)
        dates: Dates of observations
        sunshine_fraction: Daily sunshine fraction of observations (unitless)
        temperature: Daily temperature of observations (°C)
    """

    def __init__(
        self,
        latitude: ArrayType[np.floating],
        elevation: ArrayType[np.floating],
        temperature: ArrayType[np.floating],
        shortwave_radiation: ArrayType[np.floating],
        dates: Calendar,
        core_const: CoreConst = CoreConst(longwave_radiation_option="Sandoval2024"),
    ) -> None:
        """Populates key fluxes from input variables."""

        super().__init__(
            latitude=latitude,
            elevation=elevation,
            temperature=temperature,
            dates=dates,
            core_const=core_const,
            kwargs=shortwave_radiation,
        )

        # TODO - need to retrieve kwargs from validation? Maybe just drop ABC and have
        #        alternate variables in __init__

        # Sequence of calculation below taken from:
        # https://github.com/dsval/rsplash/blob/master/src/SOLAR.cpp

        # Calculate cloud free transmisivity (tau_o), unitless
        # Eq. 11, Linacre (1968); Eq. 2, Allen (1996)
        tau_o = calculate_transmissivity(
            sunshine_fraction=np.array([1.0]),
            elevation=elevation,
            coef=self.core_const.transmissivity_coef,
        )

        # Calculate realised transmisivity (tau) as the ratio of observed surface
        # shortwave downwelling radiation to top of atmosphere radiation
        # TODO - handle edge cases
        daily_sw = shortwave_radiation * self.core_const.day_seconds
        self.transmissivity = daily_sw / self.daily_solar_radiation

        # Calculate daily PPFD (ppfd_d), mol/m^2
        # TODO - although this differs markedly from the straightforward prediction from
        #        SW * swdown_to_ppfd_factor
        self.daily_ppfd = calculate_ppfd_from_tau_rd(
            transmissivity=self.transmissivity,
            daily_solar_radiation=self.daily_solar_radiation,
            swdown_to_ppfd_factor=self.core_const.swdown_to_ppfd_factor,
            visible_light_albedo=self.core_const.visible_light_albedo,
        )

        # Calculate the sunshine fraction
        self.sunshine_fraction = calculate_sunshine_fraction(
            realised_transmissivity=self.transmissivity, clear_sky_transmissivity=tau_o
        )

        # Estimate net longwave radiation (rnl), W/m^2
        # Eq. 11, Prentice et al. (1993); Eq. 5 and 6, Linacre (1968)
        self.net_longwave_radiation = calculate_net_longwave_radiation(
            sunshine_fraction=self.sunshine_fraction,
            temperature=temperature,
            coef=self.core_const.net_longwave_radiation_coef,
        )

        self.rw = calculate_rw_intermediate_from_sw(
            shortwave_radiation=shortwave_radiation,
            sunset_hour_angle=self.sunset_hour_angle,
            ru=self.ru,
            rv=self.rv,
            shortwave_albedo=self.core_const.shortwave_albedo,
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
