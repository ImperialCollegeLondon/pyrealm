"""The evap submodule provides functions and classes to calculate evaporative fluxes."""

from dataclasses import InitVar, dataclass, field
from typing import Optional, Union

import numpy as np

from pyrealm.splash.const import kCw, kG, kL, kMa, kMv, kPo, kR, kTo, kw, pir
from pyrealm.splash.solar import DailySolarFluxes
from pyrealm.utilities import check_input_shapes


@dataclass
class DailyEvapFluxes:
    """Calculate daily evaporative fluxes.

    This class calculates daily evapotranspiration fluxes given temperature and
    atmospheric pressure for daily observations and the calculated solar fluxes for
    those observations. The :meth:`~pyrealm.splash.evap.DailyEvapFluxes.estimate_aet`
    method can then be used to estimate actual evapotranspiration for observations,
    given estimates of the evaporative supply rate.

    Args:
        solar: The daily solar fluxes for the observations
        kWm: The maximum soil water capacity (mm)
        tc: The air temperature of the observations (°C)
        pa: The atmospheric pressure of the observations
    """

    solar: DailySolarFluxes
    pa: InitVar[np.ndarray]
    tc: InitVar[np.ndarray]
    kWm: np.ndarray = np.array([150.0])

    sat: np.ndarray = field(init=False)
    """Slope of saturation vapour pressure temperature curve, Pa/K"""
    lv: np.ndarray = field(init=False)
    """Enthalpy of vaporization, J/kg"""
    pw: np.ndarray = field(init=False)
    """Density of water, kg/m^3"""
    psy: np.ndarray = field(init=False)
    """Psychrometric constant, Pa/K"""
    econ: np.ndarray = field(init=False)
    """Water-to-energy conversion factor"""
    cond: np.ndarray = field(init=False)
    """Daily condensation, mm"""
    eet_d: np.ndarray = field(init=False)
    """Daily EET, mm"""
    pet_d: np.ndarray = field(init=False)
    """Daily PET, mm"""
    rx: np.ndarray = field(init=False)
    """Variable substitute, (mm/hr)/(W/m^2)"""

    def __post_init__(self, pa: np.ndarray, tc: np.ndarray) -> None:
        """Calculate invariant components of evapotranspiration.

        The post_init method calculates the invariant components of the
        evapotranspiration fluxes. Two remaining components, the intersection hour angle
        (hi), degrees and the estimated daily AET (aet_d), mm, depend on the estimated
        evaporative supply rate, which can be updated during spin up the SPLASH model.
        """

        # Slope of saturation vap press temp curve, Pa/K
        self.sat = sat_slope(tc)

        # Enthalpy of vaporization, J/kg
        self.lv = enthalpy_vap(tc)

        # Density of water, kg/m^3
        self.pw = density_h2o(tc, pa)

        # Psychrometric constant, Pa/K
        self.psy = psychro(tc, pa)

        # Calculate water-to-energy conversion (econ), m^3/J
        self.econ = self.sat / (self.lv * self.pw * (self.sat + self.psy))

        # Calculate daily condensation (cn), mm
        self.cond = (1e3) * self.econ * np.abs(self.solar.rnn_d)

        # Estimate daily EET (eet_d), mm
        self.eet_d = (1e3) * self.econ * self.solar.rn_d

        # Estimate daily PET (pet_d), mm
        self.pet_d = (1.0 + kw) * self.eet_d

        # Calculate variable substitute (rx), (mm/hr)/(W/m^2)
        self.rx = (3.6e6) * (1.0 + kw) * self.econ

    def estimate_aet(
        self, wn: np.ndarray, day_idx: Optional[int] = None, only_aet: bool = True
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Estimate actual evapotranspiration.

        This method estimates the estimated daily actual evapotranspiration (AET,
        mm/day), given estimates of the soil moisture  (wn) for observations.
        Optionally, the method can also return the the intersection hour angle (hi,
        degrees) and evaporative supply rate (sw, mm/h).

        By default, sw is expected to provide estimates for all observations across all
        days in the model, but day_idx can be set to provide sw for only one particular
        day of observations.

        Args:
            sw: The soil moisture (mm).
            day_idx: An integer giving the index of the sw values along the time axis.
            aet_only: Should the function only return AET or AET, hi and sw.

        Returns:
            An array of AET values or a tuple of arrays containing AET, hi and sw.
        """

        # Check day_idx inputs
        if day_idx is None:
            check_input_shapes(wn, self.sat)
            didx: Union[int, slice] = slice(self.sat.shape[0])
        else:
            check_input_shapes(wn, self.sat[day_idx])
            didx = day_idx

        # Calculate evaporative supply rate (sw), mm/h
        sw = kCw * wn / self.kWm

        # Validate evaporative supply rate
        if np.any(sw < 0):
            raise ValueError(
                "Evaporative supply rate contains values outside range of validity"
            )

        # Calculate the intersection hour angle (hi), degrees, guarding against np.nan
        # values from np.arccos(v > 1), setting this directly to 1
        hi_pre = (
            sw / (self.solar.rw[didx] * self.solar.rv[didx] * self.rx[didx])
            + self.solar.rnl[didx] / (self.solar.rw[didx] * self.solar.rv[didx])
            - self.solar.ru[didx] / self.solar.rv[didx]
        )
        hi = np.arccos(np.clip(hi_pre, -np.inf, 1)) / pir

        # Estimate daily AET (aet_d), mm
        aet_d = (
            (sw * np.deg2rad(hi))
            + (
                self.rx[didx]
                * self.solar.rw[didx]
                * self.solar.rv[didx]
                * (np.sin(np.deg2rad(self.solar.hn[didx])) - np.sin(np.deg2rad(hi)))
            )
            + (
                (
                    self.rx[didx] * self.solar.rw[didx] * self.solar.ru[didx]
                    - self.rx[didx] * self.solar.rnl[didx]
                )
                * (self.solar.hn[didx] - hi)
                * pir
            )
        ) * (24.0 / np.pi)

        if only_aet:
            return aet_d
        else:
            return aet_d, hi, sw


def sat_slope(tc: np.ndarray) -> np.ndarray:
    """Calculate the slope of the saturation vapour pressure curve.

    Calculates the slope of the saturation pressure temperature curve (Pa/K) following
    equation 13 of :cite:t:`allen:1998a`.

    Args:
        tc: The air temperature (°C)

    Returns
        The calculated slope.
    """
    return (
        (17.269)
        * (237.3)
        * (610.78)
        * (np.exp(tc * 17.269 / (tc + 237.3)) / ((tc + 237.3) ** 2))
    )


def enthalpy_vap(tc: np.ndarray) -> np.ndarray:
    """Calculate the enthalpy of vaporization.

    Calculates the latent heat of vaporization of water (J/Kg) as a function of
    temperature following :cite:t:`henderson-sellers:1984a`.

    Args:
        tc: Air temperature (°C)

    Returns:
        Calculated latent heat of vaporisation.
    """

    return 1.91846e6 * ((tc + 273.15) / (tc + 273.15 - 33.91)) ** 2


def elv2pres(z: np.ndarray) -> np.ndarray:
    """Calculate atmospheric pressure (Pa).

    Follows :cite:t:`allen:1998a`.

    Args:
        z: Elevation (m)

    Returns:
        Atmospheric pressure.
    """

    # TODO - replace
    return kPo * (1.0 - kL * z / kTo) ** (kG * kMa / (kR * kL))


def density_h2o(tc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Calculate the density of water.

    This function calculates thedensity of water at a given temperature and pressure
    (kg/m^3) following :cite:t:`chen:2008a`.

    Args:
        tc: Air temperature (°C)
        p: Atmospheric pressure (Pa)

    Returns:
        The calculated density of water
    """

    # TODO - merge

    # Calculate density at 1 atm (kg/m^3):
    po = 0.99983952 + (6.788260e-5) * tc
    po += -(9.08659e-6) * tc * tc
    po += (1.022130e-7) * tc * tc * tc
    po += -(1.35439e-9) * tc * tc * tc * tc
    po += (1.471150e-11) * tc * tc * tc * tc * tc
    po += -(1.11663e-13) * tc * tc * tc * tc * tc * tc
    po += (5.044070e-16) * tc * tc * tc * tc * tc * tc * tc
    po += -(1.00659e-18) * tc * tc * tc * tc * tc * tc * tc * tc

    # Calculate bulk modulus at 1 atm (bar):
    ko = 19652.17 + 148.1830 * tc
    ko += -2.29995 * tc * tc
    ko += 0.01281 * tc * tc * tc
    ko += -(4.91564e-5) * tc * tc * tc * tc
    ko += (1.035530e-7) * tc * tc * tc * tc * tc

    # Calculate temperature dependent coefficients:
    ca = 3.26138 + (5.223e-4) * tc
    ca += (1.324e-4) * tc * tc
    ca += -(7.655e-7) * tc * tc * tc
    ca += (8.584e-10) * tc * tc * tc * tc

    cb = 7.2061e-5 + -(5.8948e-6) * tc
    cb += (8.69900e-8) * tc * tc
    cb += -(1.0100e-9) * tc * tc * tc
    cb += (4.3220e-12) * tc * tc * tc * tc

    # Convert atmospheric pressure to bar (1 bar = 100000 Pa)
    pbar = (1.0e-5) * p

    pw = ko + ca * pbar + cb * pbar**2.0
    pw /= ko + ca * pbar + cb * pbar**2.0 - pbar
    pw *= (1e3) * po
    return pw


def psychro(tc: np.ndarray, p: np.ndarray) -> np.ndarray:
    r"""Calculate the psychrometric constant.

    Calculates the psychrometric constant (:math:`\lambda`, Pa/K) given the temperature
    and atmospheric pressure following :cite:t:`allen:1998a` and
    :cite:t:`tsilingiris:2008a`.

    Args:
        tc: Air temperature (°C)
        p: Atmospheric pressure (Pa)

    Returns:
        The calculated psychrometric constant
    """

    # Calculate the specific heat capacity of water, J/kg/K
    cp = specific_heat(tc)

    # Calculate latent heat of vaporization, J/kg
    lv = enthalpy_vap(tc)

    # Calculate psychrometric constant, Pa/K
    # Eq. 8, Allen et al. (1998)
    return cp * kMa * p / (kMv * lv)


def specific_heat(tc: np.ndarray) -> np.ndarray:
    """Calculate the specific heat of air.

    Calculates the specific heat of air at a constant pressure (:math:`c_{pm}`, J/kg/K)
    following :cite:t:`tsilingiris:2008a`. This equation is only valid for temperatures
    between 0 and 100 °C.

    Args:
        tc: Air temperature (°C)

    Returns:
        The specific heat of air values.
    """
    tc = np.clip(tc, 0, 100)
    cp = 1.0045714270 + (2.050632750e-3) * tc
    cp += -(1.631537093e-4) * tc * tc
    cp += (6.212300300e-6) * tc * tc * tc
    cp += -(8.830478888e-8) * tc * tc * tc * tc
    cp += (5.071307038e-10) * tc * tc * tc * tc * tc
    cp *= 1e3

    return cp