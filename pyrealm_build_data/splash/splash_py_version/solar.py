#!/usr/bin/python
#
# solar.py
#
# VERSION: 1.0-r2
# LAST UPDATED: 2016-08-19
#
# ~~~~~~~~
# license:
# ~~~~~~~~
# Copyright (C) 2016 Prentice Lab
#
# This file is part of the SPLASH model.
#
# SPLASH is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2.1 of the License, or
# (at your option) any later version.
#
# SPLASH is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SPLASH.  If not, see <http://www.gnu.org/licenses/>.
#
# ~~~~~~~~~
# citation:
# ~~~~~~~~~
# T. W. Davis, I. C. Prentice, B. D. Stocker, R. J. Whitley, H. Wang, B. J.
# Evans, A. V. Gallego-Sala, M. T. Sykes, and W. Cramer, Simple process-
# led algorithms for simulating habitats (SPLASH): Robust indices of radiation,
# evapotranspiration and plant-available moisture, Geoscientific Model
# Development, 2016 (in progress)

###############################################################################
# IMPORT MODULES:
###############################################################################
import logging

import numpy

# NOTE: For palaeoclimate studies, import separate ke, keps and komega values
#       corresponding to your study period
from const import (ke, keps, kGsc, kA, kb, kc, kd, kfFEC, kalb_vis, kalb_sw,
                   komega, pir)
from utilities import dcos
from utilities import dsin


###############################################################################
# CLASSES
###############################################################################
class SOLAR:
    """
    Name:     SOLAR
    Features: This class calculates the daily radiation fluxes.
    History:  Version 1.0-r2
              - fixed HN- equation (iss#13) [16.08.19]
              Version 1.0-r1
              - separated daily flux calculation from init [15.12.29]
              - created print vals function [16.01.29]
              - updated documentation [16.05.27]
    """
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Initialization
    # ////////////////////////////////////////////////////////////////////////
    def __init__(self, lat, elv=0.0):
        """
        Name:     SOLAR.__init__
        Inputs:   - float, latitude, degrees (lat)
                  - float, elevation, m (elv)
        Features: Initializes point-based radiation method
        """
        # Create a class logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("SOLAR class called")

        # Assign default public variables:
        self.elv = elv
        self.logger.info("elevation set to %0.3f m", elv)

        # Error handle and assign required public variables:
        if lat > 90.0 or lat < -90.0:
            self.logger.error(
                "Latitude outside range of validity, (-90 to 90)!")
            raise ValueError(
                "Latitude outside range of validity, (-90 to 90)!")
        else:
            self.logger.info("latitude set to %0.3f degrees", lat)
            self.lat = lat

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Function Definitions
    # ////////////////////////////////////////////////////////////////////////
    def berger_tls(self, n):
        """
        Name:     SOLAR.berger_tls
        Input:    int, day of year
        Output:   tuple,
                  - true anomaly, degrees
                  - true longitude, degrees
        Features: Returns true anomaly and true longitude for a given day
        Depends:  - ke
                  - komega
        Ref:      Berger, A. L. (1978), Long term variations of daily
                  insolation and quaternary climatic changes, J. Atmos. Sci.,
                  35, 2362-2367.
        """
        self.logger.debug("calculating heliocentric longitudes for day %d", n)

        # Variable substitutes:
        xee = ke**2
        xec = ke**3
        xse = numpy.sqrt(1.0 - xee)

        # Mean longitude for vernal equinox:
        xlam = (ke/2.0 + xec/8.0)*(1.0 + xse)*dsin(komega)
        xlam -= xee/4.0*(0.5 + xse)*dsin(2.0*komega)
        xlam += xec/8.0*(1.0/3.0 + xse)*dsin(3.0*komega)
        xlam *= 2.0
        xlam /= pir
        self.logger.debug("mean longitude for vernal equinox set to %f", xlam)

        # Mean longitude for day of year:
        dlamm = xlam + (n - 80.0)*(360.0/self.kN)
        self.logger.debug("mean longitude for day of year set to %f", dlamm)

        # Mean anomaly:
        anm = (dlamm - komega)
        ranm = (anm*pir)
        self.logger.debug("mean anomaly set to %f", ranm)

        # True anomaly:
        ranv = ranm
        ranv += (2.0*ke - xec/4.0)*numpy.sin(ranm)
        ranv += 5.0/4.0*xee*numpy.sin(2.0*ranm)
        ranv += 13.0/12.0*xec*numpy.sin(3.0*ranm)
        anv = ranv/pir

        # True longitude:
        my_tls = anv + komega
        if my_tls < 0:
            my_tls += 360.0
        elif my_tls > 360:
            my_tls -= 360.0
        self.logger.debug("true longitude set to %f", my_tls)

        # True anomaly:
        my_nu = (my_tls - komega)
        if my_nu < 0:
            my_nu += 360.0
        self.logger.debug("true anomaly set to %f", my_nu)

        return(my_nu, my_tls)

    def calculate_daily_fluxes(self, n, y=0, sf=1.0, tc=23.0):
        """
        Name:     SOLAR.calculate_daily_fluxes
        Input:    - int, day of the year (n)
                  - [optional] int, year (y)
                  - [optional] float, fraction of sunshine hours (sf)
                  - [optional] float, mean daily air temperature, C (tc)
        Depends:  - julian_day
                  - berger_tls
                  - dcos
                  - dsin
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 0. Validate day of year
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if n < 1 or n > 366:
            self.logger.error(
                "Day of year outside range of validity, (1 to 366)!")
            raise ValueError(
                "Day of year outside range of validity (1 to 366)!")
        else:
            self.day = n

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Calculate number of days in year (kN), days
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if y == 0:
            kN = 365
            self.year = 2001
        elif y < 0:
            self.logger.error("year set out of range")
            raise ValueError(
                "Please use a valid Julian or Gregorian calendar year")
        else:
            kN = self.julian_day((y+1), 1, 1) - self.julian_day(y, 1, 1)
            self.year = y
        self.kN = kN
        self.logger.info(
            ("calculating daily radiation fluxes for day %d of %d "
             "for year %d with sunshine fraction %f and air temperature "
             "%f Celcius") % (n, kN, self.year, sf, tc))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Calculate heliocentric longitudes (nu and lambda), degrees
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Berger (1978)
        my_nu, my_lambda = self.berger_tls(n)
        self.my_nu = my_nu
        self.my_lambda = my_lambda
        self.logger.info("true anomaly, nu, set to %f degrees", my_nu)
        self.logger.info("true lon, lambda, set to %f degrees", my_lambda)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate distance factor (dr), unitless
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Berger et al. (1993)
        kee = ke**2
        my_rho = (1.0 - kee)/(1.0 + ke*dcos(my_nu))
        dr = (1.0/my_rho)**2
        self.dr = dr
        self.logger.info("relative Earth-Sun distance, rho, set to %f", my_rho)
        self.logger.info("distance factor, dr, set to %f", dr)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Calculate declination angle (delta), degrees
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Woolf (1968)
        delta = numpy.arcsin(dsin(my_lambda)*dsin(keps))
        delta /= pir
        self.delta = delta
        self.logger.info("declination, delta, set to %f", delta)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. Calculate variable substitutes (u and v), unitless
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ru = dsin(delta)*dsin(self.lat)
        rv = dcos(delta)*dcos(self.lat)
        self.ru = ru
        self.rv = rv
        self.logger.info("variable substitute, ru, set to %f", ru)
        self.logger.info("variable substitute, rv, set to %f", rv)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6. Calculate the sunset hour angle (hs), degrees
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Eq. 3.22, Stine & Geyer (2001)
        if (ru/rv) >= 1.0:
            # Polar day (no sunset)
            self.logger.debug("polar day---no sunset")
            hs = 180.0
        elif (ru/rv) <= -1.0:
            # Polar night (no sunrise)
            self.logger.debug("polar night---no sunrise")
            hs = 0.0
        else:
            hs = -1.0*ru/rv
            hs = numpy.arccos(hs)
            hs /= pir
        self.hs = hs
        self.logger.info("sunset angle, hs, set to %f", hs)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 7. Calculate daily extraterrestrial solar radiation (ra_d), J/m^2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Eq. 1.10.3, Duffy & Beckman (1993)
        ra_d = (86400.0/numpy.pi)*kGsc*dr*(ru*pir*hs + rv*dsin(hs))
        self.ra_d = ra_d
        self.logger.info("daily ET radiation set to %f MJ/m^2", (1.0e-6)*ra_d)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 8. Calculate transmittivity (tau), unitless
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Eq. 11, Linacre (1968); Eq. 2, Allen (1996)
        tau_o = (kc + kd*sf)
        tau = tau_o*(1.0 + (2.67e-5)*self.elv)
        self.tau = tau
        self.logger.info("base transmittivity set to %f", tau_o)
        self.logger.info("transmittivity set to %f", tau)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 9. Calculate daily photosynth. photon flux density (ppfd_d), mol/m^2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ppfd_d = (1.0e-6)*kfFEC*(1.0 - kalb_vis)*tau*ra_d
        self.ppfd_d = ppfd_d
        self.logger.info("daily PPFD set to %f mol/m^2", ppfd_d)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 10. Estimate net longwave radiation (rnl), W/m^2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Eq. 11, Prentice et al. (1993); Eq. 5 and 6, Linacre (1968)
        rnl = (kb + (1.0 - kb)*sf)*(kA - tc)
        self.rnl = rnl
        self.logger.info("net longwave radiation set to %f W/m^2", rnl)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 11. Calculate variable substitute (rw), W/m^2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rw = (1.0 - kalb_sw)*tau*kGsc*dr
        self.rw = rw
        self.logger.info("variable substitute, rw, set to %f", rw)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 12. Calculate net radiation cross-over hour angle (hn), degrees
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (rnl - rw*ru)/(rw*rv) >= 1.0:
            # Net radiation negative all day
            self.logger.debug("net radiation negative all day")
            hn = 0
        elif (rnl - rw*ru)/(rw*rv) <= -1.0:
            # Net radiation positive all day
            self.logger.debug("net radiation positive all day")
            hn = 180.0
        else:
            hn = (rnl - rw*ru)/(rw*rv)
            hn = numpy.arccos(hn)
            hn /= pir
        self.hn = hn
        self.logger.info("cross-over hour angle set to %f", hn)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 13. Calculate daytime net radiation (rn_d), J/m^2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rn_d = (86400.0/numpy.pi)*(hn*pir*(rw*ru - rnl) + rw*rv*dsin(hn))
        self.rn_d = rn_d
        self.logger.info(
            "daytime net radiation set to %f MJ/m^2", (1.0e-6)*rn_d)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 14. Calculate nighttime net radiation (rnn_d), J/m^2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # fixed iss#13
        rnn_d = rw*rv*(dsin(hs) - dsin(hn))
        rnn_d += rw*ru*pir*(hs - hn)
        rnn_d -= rnl*(numpy.pi - pir*hn)
        rnn_d *= (86400.0/numpy.pi)
        self.rnn_d = rnn_d
        self.logger.info(
            "nighttime net radiation set to %f MJ/m^2", (1.0e-6)*rnn_d)

    def julian_day(self, y, m, i):
        """
        Name:     SOLAR.julian_day
        Input:    - int, year (y)
                  - int, month (i.e., 1--12) (m)
                  - int, day of month (i.e., 1--31) (i)
        Output:   float, Julian Ephemeris Day
        Features: Converts Gregorian date (year, month, day) to Julian
                  Ephemeris Day
                  * valid for dates after -4712 January 1 (i.e., jde >= 0)
        Ref:      Eq. 7.1, Meeus, J. (1991), Ch.7 "Julian Day," Astronomical
                  Algorithms
        """
        self.logger.debug("calculating Julian day")
        if m <= 2.0:
            y -= 1.0
            m += 12.0

        a = int(y/100)
        b = 2 - a + int(a/4)

        jde = int(365.25*(y + 4716)) + int(30.6001*(m + 1)) + i + b - 1524.5
        return jde

    def print_vals(self):
        """
        Name:     SOLAR.print_vals
        Inputs:   None.
        Outputs:  None.
        Features: Prints daily radiation fluxes
        """
        print("year: %d" % (self.year))
        print("day of year, n: %d" % (self.day))
        print("days in year, kN: %d" % (self.kN))
        print("true anomaly, nu: %0.6f degrees" % (self.my_nu))
        print("true lon, lambda: %0.6f degrees" % (self.my_lambda))
        print("distance factor, dr: %0.6f" % (self.dr))
        print("declination, delta: %0.6f" % (self.delta))
        print("variable substitute, ru: %0.6f" % (self.ru))
        print("variable substitute, rv: %0.6f" % (self.rv))
        print("sunset angle, hs: %0.6f degrees" % (self.hs))
        print("daily ET radiation: %0.6f MJ/m^2" % ((1.0e-6)*self.ra_d))
        print("transmittivity, tau: %0.6f" % (self.tau))
        print("daily PPFD: %0.6f mol/m^2" % (self.ppfd_d))
        print("net longwave radiation: %0.6f W/m^2" % (self.rnl))
        print("variable substitute, rw: %0.6f" % (self.rw))
        print("cross-over hour angle: %0.6f degrees" % (self.hn))
        print("daytime net radiation: %0.6f MJ/m^2" % ((1.0e-6)*self.rn_d))
        print("nighttime net radiation: %0.6f MJ/m^2" % ((1.0e-6)*self.rnn_d))

###############################################################################
# MAIN PROGRAM
###############################################################################
if __name__ == '__main__':
    # Create a root logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Instantiating logging handler and record format:
    root_handler = logging.FileHandler("solar.log")
    rec_format = "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s"
    formatter = logging.Formatter(rec_format, datefmt="%Y-%m-%d %H:%M:%S")
    root_handler.setFormatter(formatter)

    # Send logging handler to root logger:
    root_logger.addHandler(root_handler)

    # Test one-year of SPLASH:
    my_lat = 37.7
    my_elv = 142.
    my_day = 172
    my_year = 2000
    my_sf = 1.0
    my_temp = 23.0

    my_class = SOLAR(my_lat, my_elv)
    my_class.calculate_daily_fluxes(my_day, my_year, my_sf, my_temp)
