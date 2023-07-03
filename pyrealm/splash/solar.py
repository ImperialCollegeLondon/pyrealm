"""The ``solar`` submodule provides functions and classes to calculate daily solar 
radiation fluxes and other radiative values. 
"""

from dataclasses import InitVar, dataclass, field
from typing import Union

import numpy as np

from pyrealm.splash.const import (
    kA,
    kalb_sw,
    kalb_vis,
    kb,
    kc,
    kd,
    ke,
    keps,
    kfFEC,
    kGsc,
    komega,
    pir,
)
from pyrealm.splash.utilities import Calendar, CalendarDay, dcos, dsin


def calc_heliocentric_longitudes(julian_day: int, n_days: int) -> tuple[int]:
    """Calculate heliocentric longitude and anomaly.

    This function calculates the heliocentric true anomaly (degrees) and true longitude
    (degrees), given the Julian day in the year and the number of days in the year,
    following

    Berger, A. L. (1978), Long term variations of daily
            insolation and quaternary climatic changes, J. Atmos. Sci.,
            35, 2362-2367.

    Args:
        julian_day: day of year
        n_days: number of days in year

    Returns:
        nu: the true anomaly, degrees
        lambda_: true longitude, degrees
    """

    # Variable substitutes:
    xee = ke**2
    xec = ke**3
    xse = np.sqrt(1.0 - xee)

    # Mean longitude for vernal equinox:
    xlam = (
        (
            ((ke / 2.0 + xec / 8.0) * (1.0 + xse) * dsin(komega))
            - (xee / 4.0 * (0.5 + xse) * dsin(2.0 * komega))
            + (xec / 8.0 * (1.0 / 3.0 + xse) * dsin(3.0 * komega))
        )
        * 2.0
        / pir
    )

    # Mean longitude for day of year:
    dlamm = xlam + (julian_day - 80.0) * (360.0 / n_days)

    # Mean anomaly:
    ranm = (dlamm - komega) * pir

    # True anomaly:
    ranv = (
        ranm
        + ((2.0 * ke - xec / 4.0) * np.sin(ranm))
        + (5.0 / 4.0 * xee * np.sin(2.0 * ranm))
        + (13.0 / 12.0 * xec * np.sin(3.0 * ranm))
    )

    # True longitude in degrees constrained to 0 - 360
    lambda_ = ((ranv / pir) + komega) % 360

    # True anomaly in degrees constrained to 0 - 360
    nu = (lambda_ - komega) % 360

    return (nu, lambda_)


@dataclass
class DailySolarFluxes:
    """Calculate daily solar fluxes.

    This dataclass takes arrays describing the latitude, elevation, sunshine fraction
    and mean daily temperature for observations and then calculates key radiation fluxes
    given a Calendar object providing the Julian day of the observations and the year
    and number of days in the year.
    """

    lat: InitVar[np.ndarray]
    """Latitude of observations, degrees"""
    elv: InitVar[np.ndarray]
    """Elevation of observations, metres"""
    dates: Union[Calendar, CalendarDay]
    sf: InitVar[np.ndarray]
    """Daily sunshine fraction of observations, unitless"""
    tc: InitVar[np.ndarray]
    """Daily temperature of observations, °C"""

    nu: np.ndarray = field(init=False)
    """True heliocentric anomaly, degrees"""
    lambda_: np.ndarray = field(init=False)
    """True heliocentric longitude, degrees"""
    dr: np.ndarray = field(init=False)
    """Distance factor, -"""
    delta: np.ndarray = field(init=False)
    """Declination angle, degrees"""
    ru: np.ndarray = field(init=False)
    """Intermediate variable, unitless"""
    rv: np.ndarray = field(init=False)
    """Intermediate variable, unitless"""
    hs: np.ndarray = field(init=False)
    """Sunset hour angle, degrees"""
    ra_d: np.ndarray = field(init=False)
    """Daily extraterrestrial solar radiation, J/m^2"""
    tau: np.ndarray = field(init=False)
    """Transmittivity, unitless"""
    ppfd_d: np.ndarray = field(init=False)
    """Daily PPFD, mol/m^2"""
    rnl: np.ndarray = field(init=False)
    """Net longwave radiation, W/m^2"""
    rw: np.ndarray = field(init=False)
    """Intermediate variable,  W/m^2"""
    hn: np.ndarray = field(init=False)
    """Net radiation cross-over hour angle, degrees"""
    rn_d: np.ndarray = field(init=False)
    """Daytime net radiation, J/m^2"""
    rnn_d: np.ndarray = field(init=False)
    """Nighttime net radiation (rnn_d), J/m^2"""

    def __post_init__(self, lat, elv, sf, tc):
        """Populates key fluxes from input variables."""

        # Calculate heliocentric longitudes (nu and lambda), Berger (1978)
        self.nu, self.lambda_ = calc_heliocentric_longitudes(
            self.dates.julian_day, self.dates.days_in_year
        )

        # Calculate distance factor (dr), Berger et al. (1993)
        self.dr = (1.0 / ((1.0 - ke**2) / (1.0 + ke * dcos(self.nu)))) ** 2

        # Calculate declination angle (delta), Woolf (1968)
        self.delta = np.arcsin(dsin(self.lambda_) * dsin(keps)) / pir

        # Calculate variable substitutes (u and v), unitless
        self.ru = dsin(self.delta) * dsin(lat)
        self.rv = dcos(self.delta) * dcos(lat)

        # Calculate the sunset hour angle (hs), Eq. 3.22, Stine & Geyer (2001)
        self.hs = np.arccos(-1.0 * np.clip(self.ru / self.rv, -1.0, 1.0)) / pir

        # Calculate daily extraterrestrial solar radiation (ra_d), J/m^2
        # Eq. 1.10.3, Duffy & Beckman (1993)
        self.ra_d = (
            (86400.0 / np.pi)
            * kGsc
            * self.dr
            * (self.ru * pir * self.hs + self.rv * dsin(self.hs))
        )

        # Calculate transmittivity (tau), unitless
        # Eq. 11, Linacre (1968); Eq. 2, Allen (1996)
        self.tau = (kc + kd * sf) * (1.0 + (2.67e-5) * elv)

        # Calculate daily PPFD (ppfd_d), mol/m^2
        self.ppfd_d = (1.0e-6) * kfFEC * (1.0 - kalb_vis) * self.tau * self.ra_d

        # Estimate net longwave radiation (rnl), W/m^2
        # Eq. 11, Prentice et al. (1993); Eq. 5 and 6, Linacre (1968)
        self.rnl = (kb + (1.0 - kb) * sf) * (kA - tc)

        # Calculate variable substitute (rw), W/m^2
        self.rw = (1.0 - kalb_sw) * self.tau * kGsc * self.dr

        # Calculate net radiation cross-over hour angle (hn), degrees
        self.hn = (
            np.arccos(
                np.clip((self.rnl - self.rw * self.ru) / (self.rw * self.rv), -1.0, 1.0)
            )
            / pir
        )

        # Calculate daytime net radiation (rn_d), J/m^2
        self.rn_d = (86400.0 / np.pi) * (
            self.hn * pir * (self.rw * self.ru - self.rnl)
            + self.rw * self.rv * dsin(self.hn)
        )

        # Calculate nighttime net radiation (rnn_d), J/m^2
        self.rnn_d = (
            (self.rw * self.rv * (dsin(self.hs) - dsin(self.hn)))
            + (self.rw * self.ru * pir * (self.hs - self.hn))
            - (self.rnl * (np.pi - pir * self.hn))
        ) * (86400.0 / np.pi)
