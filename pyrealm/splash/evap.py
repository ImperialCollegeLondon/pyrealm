"""The ``evap`` submodule provides functions and classes to calculate evaporative
fluxes.
"""  # noqa: D205

from dataclasses import InitVar, dataclass, field

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.hygro import (
    calc_enthalpy_vaporisation,
    calc_psychrometric_constant,
    calc_saturation_vapour_pressure_slope,
)
from pyrealm.core.time_series import broadcast_time
from pyrealm.core.utilities import check_input_shapes
from pyrealm.core.water import calc_density_h2o
from pyrealm.splash.solar import DailySolarFluxes


@dataclass
class DailyEvapFluxes:
    """Calculate daily evaporative fluxes.

    This class calculates daily evaporative fluxes given temperature and atmospheric
    pressure for daily observations and the calculated solar fluxes for those
    observations. The class attributes provide components of those fluxes that depend
    only on the initial data and are not dependent on the previous model state when
    iterating over time.

    Two remaining components, the intersection hour angle (hi, degrees) and the
    estimated daily AET (aet_d, mm), depend on the soil moisture from the previous day
    and so must be calculated on a daily basis during either the spin process of the
    SPLASH model to estimate initial equilibrium soil moisture, or the daily calculation
    of soil moisture along a time series. These quantities are estimated using the
    :meth:`~pyrealm.splash.evap.DailyEvapFluxes.estimate_aet` method, given an estimate
    of soil moisture from the preceeding day.

    Args:
        solar: A :class:`~pyrealm.splash.solar.DailySolarFluxes` instance from which to
            calculate evaporative fluxes. See the class definition for the flux
            variables and units provided.
        kWm: The maximum soil water capacity (mm).
        tc: The air temperature of the observations (°C).
        pa: The atmospheric pressure of the observations (Pa).
        core_const: An instance of CoreConst.
    """

    solar: DailySolarFluxes
    pa: InitVar[NDArray]
    tc: InitVar[NDArray]
    kWm: NDArray[np.float64] = field(default_factory=lambda: np.array([150.0]))
    core_const: CoreConst = field(default_factory=lambda: CoreConst())

    sat: NDArray[np.float64] = field(init=False)
    """Slope of saturation vapour pressure temperature curve, Pa/K"""
    lv: NDArray[np.float64] = field(init=False)
    """Enthalpy of vaporization, J/kg"""
    pw: NDArray[np.float64] = field(init=False)
    """Density of water, kg/m^3"""
    psy: NDArray[np.float64] = field(init=False)
    """Psychrometric constant, Pa/K"""
    econ: NDArray[np.float64] = field(init=False)
    """Water-to-energy conversion factor"""
    cond: NDArray[np.float64] = field(init=False)
    """Daily condensation, mm"""
    eet_d: NDArray[np.float64] = field(init=False)
    """Daily equilibrium evapotranspiration (EET), mm"""
    pet_d: NDArray[np.float64] = field(init=False)
    """Daily potential evapotranspiration (PET), mm"""
    rx: NDArray[np.float64] = field(init=False)
    """Variable substitute, (mm/hr)/(W/m^2)"""

    def __post_init__(self, pa: NDArray[np.float64], tc: NDArray[np.float64]) -> None:
        """Calculate invariant components of evapotranspiration.

        The post_init method calculates the components of the evaporative fluxes that
        depend only on the initial data and are not dependent on the previous model
        state when iterating over time.
        """

        try:
            self.shape: tuple = check_input_shapes(pa, tc, shape=self.solar.shape)
            """The array shape of the input variables"""
        except ValueError:
            msg = (
                "The shape of DailyEvapFluxes inputs are inconsistent with each other "
                "or the DailySolarFluxes data"
            )
            raise ValueError(msg)

        # Broadcast along the time axis (necessary for the indexing in estimate_aet)
        pa = broadcast_time(pa, self.shape)
        tc = broadcast_time(tc, self.shape)

        # Slope of saturation vap press temp curve, Pa/K
        self.sat = calc_saturation_vapour_pressure_slope(tc)

        # Enthalpy of vaporization, J/kg
        self.lv = calc_enthalpy_vaporisation(tc)

        # Density of water, kg/m^3
        self.pw = calc_density_h2o(tc, pa, core_const=self.core_const)

        # Psychrometric constant, Pa/K
        self.psy = calc_psychrometric_constant(tc, pa, core_const=self.core_const)

        # Calculate water-to-energy conversion (econ), m^3/J
        self.econ = self.sat / (self.lv * self.pw * (self.sat + self.psy))

        # Calculate daily condensation (cn), mm
        self.cond = (1e3) * self.econ * np.abs(self.solar.nighttime_net_radiation)

        # Estimate daily equilibrium evapotranspiration (eet_d), mm
        self.eet_d = (1e3) * self.econ * self.solar.daytime_net_radiation

        # Estimate daily potential evapotranspiration (pet_d), mm
        self.pet_d = (1.0 + self.core_const.k_w) * self.eet_d

        # Calculate variable substitute (rx), (mm/hr)/(W/m^2)
        self.rx = (3.6e6) * (1.0 + self.core_const.k_w) * self.econ

    def estimate_aet(
        self, wn: NDArray[np.float64], day_idx: int | None = None, only_aet: bool = True
    ) -> NDArray[np.float64] | tuple[NDArray, NDArray, NDArray]:
        """Estimate actual evapotranspiration.

        This method estimates the daily actual evapotranspiration (AET, mm/day), given
        estimates of the soil moisture  (``wn``) for observations. Optionally, the
        method can also return the the intersection hour angle (``hi``, degrees) and
        evaporative supply rate (``sw``, mm/h).

        By default, ``wn`` is expected to provide estimates for all observations across
        all days in the model, but ``day_idx`` can be set to indicate that ``wn`` is
        providing the soil moisture for one specific day across the observations.

        Args:
            wn: The soil moisture (mm).
            day_idx: An integer giving the index of the provided ``wn`` values along the
                time axis.
            only_aet: Should the function only return AET or AET, ``hi`` and ``sw``.

        Returns:
            An array of AET values or a tuple of arrays containing AET, ``hi`` and
            ``sw``.
        """

        # Check day_idx inputs and create the indexing object `didx`, used to either
        # subset the calculations to particular request days or use the entire array of
        # soil moisture. The slice here is used to programatically select `array[:]`.
        if day_idx is None:
            splash_shape = self.shape
            didx: int | slice = slice(self.shape[0])
        else:
            splash_shape = self.shape[1:]
            didx = day_idx
        try:
            check_input_shapes(wn, shape=splash_shape)
        except ValueError:
            msg = "The shape of wn does not match the existing SPLASH model data"
            raise ValueError(msg)

        # Calculate evaporative supply rate (sw), mm/h
        sw = self.core_const.k_Cw * wn / self.kWm

        # Validate evaporative supply rate
        if np.any(sw < 0):
            raise ValueError(
                "Evaporative supply rate contains values outside range of validity"
            )

        # Calculate the intersection hour angle (hi), degrees, guarding against np.nan
        # values from np.arccos(v > 1), setting this directly to 1
        hi_pre = (
            sw / (self.solar.rw[didx] * self.solar.rv[didx] * self.rx[didx])
            + self.solar.net_longwave_radiation[didx]
            / (self.solar.rw[didx] * self.solar.rv[didx])
            - self.solar.ru[didx] / self.solar.rv[didx]
        )
        hi = np.arccos(np.clip(hi_pre, -np.inf, 1)) / self.core_const.k_pir

        # Estimate daily actual evapotranspiration (aet_d), mm
        aet_d = (
            (sw * np.deg2rad(hi))
            + (
                self.rx[didx]
                * self.solar.rw[didx]
                * self.solar.rv[didx]
                * (
                    np.sin(np.deg2rad(self.solar.crossover_hour_angle[didx]))
                    - np.sin(np.deg2rad(hi))
                )
            )
            + (
                (
                    self.rx[didx] * self.solar.rw[didx] * self.solar.ru[didx]
                    - self.rx[didx] * self.solar.net_longwave_radiation[didx]
                )
                * (self.solar.crossover_hour_angle[didx] - hi)
                * self.core_const.k_pir
            )
        ) * (24.0 / np.pi)

        if only_aet:
            return aet_d
        else:
            return aet_d, hi, sw
