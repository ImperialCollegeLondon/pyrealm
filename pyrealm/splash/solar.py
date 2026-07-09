"""The ``solar`` submodule provides the DailySolarFluxes class, used to calculate daily
solar radiation fluxes for observations.
"""  # noqa: D205

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


class DailySolarFluxes:
    """Calculate daily solar fluxes.

    This dataclass takes arrays describing the latitude, elevation and mean daily
    temperature for observations and then calculates key radiation fluxes given a
    Calendar object providing the Julian day of the observations and the year and number
    of days in the year.

    There are two options for providing radiation inputs:

    * Sunshine fraction:  this was used in the original SPLASH implementation
      :cite:p:`davis:2017a` and is provided as the radiation input variable in - for
      example - the CRU climate datasets.

    * Shortwave radiation: the updated SPLASH v2 :cite:p:`sandoval:2024a` updated the
      calculation of solar fluxes to use data on shortwave downwelling radiation. This
      is provided as a radiation input in many other datasets, such as FluxNET sites and
      ERA5.

    The parameters for the calculation of net longwave radiation differs between these
    two options: see
    :attr:`~pyrealm.constants.core_const.CoreConst.net_longwave_radiation_coef_sw` and
    :attr:`~pyrealm.constants.core_const.CoreConst.net_longwave_radiation_coef_sf`.



    The first dimension for the array inputs must correspond to the length of the time
    series passed in using the ``dates`` argument. If xarray inputs are used, ``dates``
    should also be initialised using xarray inputs to ensure this.

    Args:
        latitude: The Latitude of observations (degrees)
        elevation: Elevation of observations (metres)
        dates: Dates of observations
        sunshine_fraction: Daily sunshine fraction of observations (unitless)
        shortwave_radiation: Daily downwelling shortwave radiation (W m-2)
        temperature: Daily temperature of observations (°C)
        core_const: Core constants
    """

    def __init__(
        self,
        latitude: ArrayType[np.floating],
        elevation: ArrayType[np.floating],
        temperature: ArrayType[np.floating],
        dates: Calendar,
        core_const: CoreConst = CoreConst(),
        sunshine_fraction: ArrayType[np.floating] | None = None,
        shortwave_radiation: ArrayType[np.floating] | None = None,
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
        self.shortwave_radiation: NDArray[np.floating]
        r"""The downwelling shortwave radiation (SW, W m-2)"""
        self.sunshine_fraction: NDArray[np.floating]
        r"""Sunshine fraction (:math:`s_f`, unitless)"""
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

        if (sunshine_fraction is None) == (shortwave_radiation is None):
            raise ValueError("Provide one of sunshine_fraction or shortwave_radiation")

        # Get a single array object for validation
        if sunshine_fraction is not None:
            radiation_input = sunshine_fraction
        elif shortwave_radiation is not None:
            radiation_input = shortwave_radiation

        # Convert any xr.DataArrays to numpy arrays
        self.dims = get_common_dims(latitude, elevation, temperature, radiation_input)
        latitude, elevation, temperature, radiation_input = xarray_inputs(
            latitude, elevation, temperature, radiation_input, dims=self.dims
        )

        # Validate the inputs
        self.shape = check_input_shapes(
            latitude, elevation, temperature, radiation_input
        )
        """The array shape of the input variables"""

        # Assign radiation variable back to the appropriate attribute
        if sunshine_fraction is not None:
            self.sunshine_fraction = radiation_input
        else:
            self.shortwave_radiation = radiation_input

        # Check the data arrays match onto the times.
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

        # Calculate intermediate values ru, rv
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

        # Calculate clear sky transmisivity (tau_o), unitless
        self.clear_sky_transmissivity = calculate_transmissivity(
            sunshine_fraction=np.array([1.0]),
            elevation=elevation,
            coef=self.core_const.transmissivity_coef,
        )

        # Now use the appropriate method for populating transmissivity, rw, daily_ppfd
        # and net_longwave_radiation
        if sunshine_fraction is not None:
            self._calculate_solar_fluxes_from_sunshine_fraction(
                temperature=temperature, elevation=elevation
            )
        else:
            self._calculate_solar_fluxes_from_shortwave_radiation(
                temperature=temperature, elevation=elevation
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

    def _calculate_solar_fluxes_from_sunshine_fraction(
        self, temperature: NDArray[np.floating], elevation: NDArray[np.floating]
    ) -> None:
        """Populate flux attributes from sunshine fraction.

        Sets transmissivity, rw, daily_ppfd, and net_longwave_radiation following
        :cite:`davis:2017a`.

        Args:
            temperature: Daily temperature of observations (°C)
            elevation: Elevation of observations (metres)
        """

        # Calculate transmittivity (tau), unitless
        self.transmissivity = calculate_transmissivity(
            sunshine_fraction=self.sunshine_fraction,
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
        self.net_longwave_radiation = calculate_net_longwave_radiation(
            sunshine_fraction=self.sunshine_fraction,
            temperature=temperature,
            coef=self.core_const.net_longwave_radiation_coef_sf,
        )

    def _calculate_solar_fluxes_from_shortwave_radiation(
        self, temperature: NDArray[np.floating], elevation: NDArray[np.floating]
    ) -> None:
        """Populate flux attributes from shortwave radiation.

        Sets transmissivity, rw, daily_ppfd, and net_longwave_radiation following
        :cite:`sandoval:2024a`

        Args:
            temperature: Daily temperature of observations (°C)
            elevation: Elevation of observations (metres)
        """

        # Sequence of calculation below taken from:
        # https://github.com/dsval/rsplash/blob/master/src/SOLAR.cpp

        # Calculate realised transmisivity (tau) as the ratio of observed surface
        # shortwave downwelling radiation to top of atmosphere radiation
        # TODO - check: the behaviour when SW > incoming is capped at tau_o but there
        #        value > tau_o are perfectly possible.
        daily_sw = self.shortwave_radiation * self.core_const.day_seconds
        ratio = daily_sw / self.daily_solar_radiation
        self.transmissivity = np.where(ratio <= 1, ratio, self.clear_sky_transmissivity)

        # Calculate daily PPFD (ppfd_d), mol/m^2
        # TODO - this differs markedly from the standard prediction from
        #        SW * swdown_to_ppfd_factor
        self.daily_ppfd = calculate_ppfd_from_tau_rd(
            transmissivity=self.transmissivity,
            daily_solar_radiation=self.daily_solar_radiation,
            swdown_to_ppfd_factor=self.core_const.swdown_to_ppfd_factor,
            visible_light_albedo=self.core_const.visible_light_albedo,
        )

        # Calculate the resulting sunshine fraction
        self.sunshine_fraction = calculate_sunshine_fraction(
            realised_transmissivity=self.transmissivity,
            clear_sky_transmissivity=self.clear_sky_transmissivity,
        )

        # Estimate net longwave radiation (rnl), W/m^2
        # Parameterisation of Sandoval et al 2024.
        self.net_longwave_radiation = calculate_net_longwave_radiation(
            sunshine_fraction=self.sunshine_fraction,
            temperature=temperature,
            coef=self.core_const.net_longwave_radiation_coef_sw,
        )

        self.rw = calculate_rw_intermediate_from_sw(
            shortwave_radiation=self.shortwave_radiation,
            sunset_hour_angle=self.sunset_hour_angle,
            ru=self.ru,
            rv=self.rv,
            shortwave_albedo=self.core_const.shortwave_albedo,
        )
