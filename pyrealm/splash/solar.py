"""The ``solar`` submodule provides functions and classes to calculate daily solar
radiation fluxes and other radiative values.
"""  # noqa D204

from dataclasses import InitVar, dataclass, field

import numpy as np
from numpy.typing import NDArray

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
from pyrealm.splash.utilities import Calendar
from pyrealm.utilities import check_input_shapes


def calc_heliocentric_longitudes(
    julian_day: NDArray, n_days: NDArray
) -> tuple[NDArray, NDArray]:
    """Calculate heliocentric longitude and anomaly.

    This function calculates the heliocentric true anomaly (``nu``, degrees) and true
    longitude (``lambda_``, degrees), given the Julian day in the year and the number of
    days in the year, following :cite:t:`berger:1978a`.

    Args:
        julian_day: day of year
        n_days: number of days in year

    Returns:
        A tuple of arrays containing ``nu`` and ``lambda_``.
    """

    # Variable substitutes:
    xee = ke**2
    xec = ke**3
    xse = np.sqrt(1.0 - xee)

    # Mean longitude for vernal equinox:
    xlam = (
        (
            ((ke / 2.0 + xec / 8.0) * (1.0 + xse) * np.sin(np.deg2rad(komega)))
            - (xee / 4.0 * (0.5 + xse) * np.sin(np.deg2rad(2.0 * komega)))
            + (xec / 8.0 * (1.0 / 3.0 + xse) * np.sin(np.deg2rad(3.0 * komega)))
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

    nu: NDArray = field(init=False)
    """True heliocentric anomaly, degrees"""
    lambda_: NDArray = field(init=False)
    """True heliocentric longitude, degrees"""
    dr: NDArray = field(init=False)
    """Distance factor, -"""
    delta: NDArray = field(init=False)
    """Declination angle, degrees"""
    ru: NDArray = field(init=False)
    """Intermediate variable, unitless"""
    rv: NDArray = field(init=False)
    """Intermediate variable, unitless"""
    hs: NDArray = field(init=False)
    """Sunset hour angle, degrees"""
    ra_d: NDArray = field(init=False)
    """Daily extraterrestrial solar radiation, J/m^2"""
    tau: NDArray = field(init=False)
    """Transmittivity, unitless"""
    ppfd_d: NDArray = field(init=False)
    """Daily PPFD, mol/m^2"""
    rnl: NDArray = field(init=False)
    """Net longwave radiation, W/m^2"""
    rw: NDArray = field(init=False)
    """Intermediate variable,  W/m^2"""
    hn: NDArray = field(init=False)
    """Net radiation cross-over hour angle, degrees"""
    rn_d: NDArray = field(init=False)
    """Daytime net radiation, J/m^2"""
    rnn_d: NDArray = field(init=False)
    """Nighttime net radiation (rnn_d), J/m^2"""

    def __post_init__(
        self, lat: NDArray, elv: NDArray, sf: NDArray, tc: NDArray
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
        dr = (1.0 / ((1.0 - ke**2) / (1.0 + ke * np.cos(np.deg2rad(nu))))) ** 2

        # Calculate declination angle (delta), Woolf (1968)
        delta = np.arcsin(np.sin(np.deg2rad(lambda_)) * np.sin(np.deg2rad(keps))) / pir

        # The nu, lambda_, dr and delta attributes are all one dimensional arrays
        # calcualted from the Calendar along the first axis of any inputs. These need to
        # be broadcastable to the shape of the other inputs. The expand_dims variable
        # gets a list of the axes to expand onto - which will be an empty list when
        # ndim=1, leaving the targets unchanged.
        expand_dims = list(np.arange(1, elv.ndim))
        self.nu = np.expand_dims(nu, axis=expand_dims)
        self.lambda_ = np.expand_dims(lambda_, axis=expand_dims)
        self.dr = np.expand_dims(dr, axis=expand_dims)
        self.delta = np.expand_dims(delta, axis=expand_dims)

        # Calculate variable substitutes (u and v), unitless
        self.ru = np.sin(np.deg2rad(self.delta)) * np.sin(np.deg2rad(lat))
        self.rv = np.cos(np.deg2rad(self.delta)) * np.cos(np.deg2rad(lat))

        # Calculate the sunset hour angle (hs), Eq. 3.22, Stine & Geyer (2001)
        self.hs = np.arccos(-1.0 * np.clip(self.ru / self.rv, -1.0, 1.0)) / pir

        # Calculate daily extraterrestrial solar radiation (ra_d), J/m^2
        # Eq. 1.10.3, Duffy & Beckman (1993)
        self.ra_d = (
            (86400.0 / np.pi)
            * kGsc
            * self.dr
            * (self.ru * pir * self.hs + self.rv * np.sin(np.deg2rad(self.hs)))
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
            + self.rw * self.rv * np.sin(np.deg2rad(self.hn))
        )

        # Calculate nighttime net radiation (rnn_d), J/m^2
        self.rnn_d = (
            (
                self.rw
                * self.rv
                * (np.sin(np.deg2rad(self.hs)) - np.sin(np.deg2rad(self.hn)))
            )
            + (self.rw * self.ru * pir * (self.hs - self.hn))
            - (self.rnl * (np.pi - pir * self.hn))
        ) * (86400.0 / np.pi)
