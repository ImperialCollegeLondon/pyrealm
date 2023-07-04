"""The evap submodule provides functions and classes to calculate evaporative fluxes."""

from dataclasses import InitVar, dataclass, field

import numpy as np

from pyrealm.splash.const import kG, kL, kMa, kMv, kPo, kR, kTo, kw, pir
from pyrealm.splash.solar import DailySolarFluxes
from pyrealm.splash.utilities import dsin


@dataclass
class DailyEvapFluxes:
    """
    Name:     EVAP
    Features: This class calculates daily radiation and evapotranspiration
              quantities:
              - PPFD (ppfd_d), mol/m^2/day
              - EET (eet_d), mm/day
              - PET (pet_d), mm/day
              - AET (aet_d), mm/day
              - condensation (cn), mm/day
    Version:  1.0.0-dev
              - replaced radiation methods with SOLAR class [15.12.29]
              - implemented logging [15.12.29]
              - created specific heat equation for Tc 0--100 deg C [16.09.09]
    """

    solar: DailySolarFluxes
    pa: InitVar[np.ndarray]
    tc: InitVar[np.ndarray]

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
    hi: np.ndarray = field(init=False)
    """Intersection hour angle (hi), degrees"""
    aet_d: np.ndarray = field(init=False)
    """Daily AET (aet_d), mm"""

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

    def estimate_hi_and_aet(self, sw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """This method estimates the intersection hour angle
        (hi), degrees and the estimated daily AET (aet_d), mm, given the estimated
        evaporative supply rate for observations."""

        # Validate evaporative supply rate
        if np.any(sw < 0):
            raise ValueError(
                "Evaporative supply rate contains values outside range of validity"
            )

        # Calculate the intersection hour angle (hi), degrees, guarding against np.nan
        # values from np.arccos(v > 1), setting this directly to 1
        hi_pre = (
            sw / (self.solar.rw * self.solar.rv * self.rx)
            + self.solar.rnl / (self.solar.rw * self.solar.rv)
            - self.solar.ru / self.solar.rv
        )
        self.hi = np.arccos(np.clip(hi_pre, -np.inf, 1)) / pir

        # Estimate daily AET (aet_d), mm
        self.aet_d = (
            (sw * self.hi * pir)
            + (
                self.rx
                * self.solar.rw
                * self.solar.rv
                * (dsin(self.solar.hn) - dsin(self.hi))
            )
            + (
                (self.rx * self.solar.rw * self.solar.ru - self.rx * self.solar.rnl)
                * (self.solar.hn - self.hi)
                * pir
            )
        ) * (24.0 / np.pi)


def sat_slope(tc: np.ndarray) -> np.ndarray:
    """
    Name:     EVAP.sat_slope
    Input:    float, air temperature (tc), degrees C
    Output:   float, slope of sat vap press temp curve (s)
    Features: Calculates the slope of the sat pressure temp curve, Pa/K
    Ref:      Eq. 13, Allen et al. (1998)
    """
    s = (
        (17.269)
        * (237.3)
        * (610.78)
        * (np.exp(tc * 17.269 / (tc + 237.3)) / ((tc + 237.3) ** 2))
    )
    # logger.debug("calculating temperature dependency at %f degrees", tc)
    return s


def enthalpy_vap(tc: np.ndarray) -> np.ndarray:
    """
    Name:     EVAP.enthalpy_vap
    Input:    float, air temperature (tc), degrees C
    Output:   float, latent heat of vaporization
    Features: Calculates the enthalpy of vaporization, J/kg
    Ref:      Eq. 8, Henderson-Sellers (1984)
    """
    # logger.debug("calculating temperature dependency at %f degrees", tc)
    return 1.91846e6 * ((tc + 273.15) / (tc + 273.15 - 33.91)) ** 2


def elv2pres(z: np.ndarray) -> np.ndarray:
    """
    Name:     EVAP.elv2pres
    Input:    float, elevation above sea level (z), m
    Output:   float, atmospheric pressure, Pa
    Features: Calculates atm. pressure for a given elevation
    Depends:  Global constants
                - kPo
                - kTo
                - kL
                - kMa
                - kG
                - kR
    Ref:      Allen et al. (1998)
    """
    # logger.debug("estimating atmospheric pressure at %f m", z)
    p = kPo * (1.0 - kL * z / kTo) ** (kG * kMa / (kR * kL))
    return p


def density_h2o(tc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Name:     EVAP.density_h2o
    Input:    - float, air temperature (tc), degrees C
                - float, atmospheric pressure (p), Pa
    Output:   float, density of water, kg/m^3
    Features: Calculates density of water at a given temperature and
                pressure
    Ref:      Chen et al. (1977)
    """

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
    """
    Name:     EVAP.psychro
    Input:    - float, air temperature (tc), degrees C
                - float, atm. pressure (p), Pa
    Output:   float, psychrometric constant, Pa/K
    Features: Calculates the psychrometric constant for a given temperature
                and pressure
    Depends:  Global constants:
                - kMa
                - kMv
    Refs:     Allen et al. (1998); Tsilingiris (2008)
    """
    # logger.debug(
    #     (
    #         "calculating psychrometric constant at temperature %f Celcius "
    #         "and pressure %f Pa"
    #     )
    #     % (tc, p)
    # )

    # Calculate the specific heat capacity of water, J/kg/K
    cp = specific_heat(tc)
    # logger.info("specific heat capacity calculated as %f J/kg/K", cp)

    # Calculate latent heat of vaporization, J/kg
    lv = enthalpy_vap(tc)
    # logger.info("enthalpy of vaporization calculated as %f MJ/kg", (1e-6) * lv)

    # Calculate psychrometric constant, Pa/K
    # Eq. 8, Allen et al. (1998)
    return cp * kMa * p / (kMv * lv)


def specific_heat(tc: np.ndarray) -> np.ndarray:
    """
    Name:     EVAP.specific_heat
    Inputs:   float, air tempearture, deg C (tc)
    Outputs:  float, specific heat of air, J/kg/K
    Features: Calculates the specific heat of air at a constant pressure;
                NOTE: this equation is only valid for temperatures between 0
                and 100 deg C
    Ref:      Eq. 47, Tsilingiris (2008)
    """
    tc = np.clip(tc, 0, 100)
    cp = 1.0045714270 + (2.050632750e-3) * tc
    cp += -(1.631537093e-4) * tc * tc
    cp += (6.212300300e-6) * tc * tc * tc
    cp += -(8.830478888e-8) * tc * tc * tc * tc
    cp += (5.071307038e-10) * tc * tc * tc * tc * tc
    cp *= 1e3

    return cp
