#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# splash.py
#
# VERSION: 1.0
# LAST UPDATED: 2016-02-19
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

from const import kCw, kWm
from evap import EVAP


###############################################################################
# CLASSES
###############################################################################
class SPLASH:
    """
    Name:     SPLASH
    Features: This class updates daily quantities of radiation,
              evapotranspiration, soil moisture and runoff based on SPLASH.
    History:  Version 1.0
              - changed xrange to range for Python 2/3 compatability [16.02.05]
    """
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Initialization
    # ////////////////////////////////////////////////////////////////////////
    def __init__(self, lat, elv):
        """
        Name:     SPLASH.__init__
        Input:    - float, latitude, degrees (lat)
                  - float, elevation, meters (elv)
        """
        # Create a class logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("SPLASH class called")

        # Error handle and assign required public variables:
        self.elv = elv
        self.logger.info("elevation set to %f m", elv)

        if lat > 90.0 or lat < -90.0:
            self.logger.error(
                "Latitude outside range of validity, (-90 to 90)!")
            raise ValueError(
                "Latitude outside range of validity, (-90 to 90)!")
        else:
            self.logger.info("latitude set to %0.3f degrees", lat)
            self.lat = lat

        # Create EVAP class:
        try:
            self.evap = EVAP(lat, elv)
        except:
            self.logger.exception("failed to Initialize EVAP class")
        else:
            self.logger.debug("initialized EVAP class")

        # Initialize daily status variables:
        self.ho = 0.      # daily solar irradiation, J/m2
        self.hn = 0.      # daily net radiation, J/m2
        self.ppfd = 0.    # daily PPFD, mol/m2
        self.cond = 0.    # daily condensation water, mm
        self.wn = 0.      # daily soil moisture, mm
        self.precip = 0.  # daily precipitation, mm
        self.ro = 0.      # daily runoff, mm
        self.eet = 0.     # daily equilibrium ET, mm
        self.pet = 0.     # daily potential ET, mm
        self.aet = 0.     # daily actual ET, mm
        self.wn_vec = numpy.array([])  # daily soil moisture array

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Function Definitions
    # ////////////////////////////////////////////////////////////////////////
    def spin_up(self, d):
        """
        Name:     SPLASH.spin
        Input:    - DATA class, (d)
                  - bool, (to_write)
        Output:   None.
        Features: Spins up the daily soil moisture, creating a daily soil
                  moisture vector (wn_vec) and previous day's soil moisture
                  value (wn).
        Depends:  quick_run
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Create a soil moisture array:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if isinstance(d.num_lines, list):
            n = d.num_lines[0]
            if (numpy.array(d.num_lines) == n).all():
                wn_vec = numpy.zeros((n,))
            else:
                self.logger.error(
                    "Inconsistent number of lines read from DATA class!")
                raise IndexError(
                    "Inconsistent number of lines read from DATA class!")
        else:
            n = d.num_lines
            wn_vec = numpy.zeros((n,))
        self.logger.info(
            "Created soil moisture array of length %d", len(wn_vec))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Run one year:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.logger.info("running first year of spin-up")
        for i in range(n):
            # Get preceding soil moisture status:
            if i == 0:
                wn = wn_vec[-1]
            else:
                wn = wn_vec[i-1]

            # Calculate soil moisture and runoff:
            sm, ro = self.quick_run(n=i+1,
                                    y=d.year,
                                    wn=wn,
                                    sf=d.sf_vec[i],
                                    tc=d.tair_vec[i],
                                    pn=d.pn_vec[i])
            wn_vec[i] = sm
        self.logger.info("completed first year")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate change in starting soil moisture:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        start_sm = wn_vec[0]
        end_sm, ro = self.quick_run(n=1,
                                    y=d.year,
                                    wn=wn_vec[-1],
                                    sf=d.sf_vec[0],
                                    tc=d.tair_vec[0],
                                    pn=d.pn_vec[0])
        diff_sm = numpy.abs(end_sm - start_sm)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Equilibrate:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.logger.info("equilibrating daily soil moisture")
        spin_count = 1
        while diff_sm > 1.0:
            self.logger.info("iteration: %d", spin_count)
            for i in range(n):
                # Get preceding soil moisture status:
                if i == 0:
                    wn = wn_vec[-1]
                else:
                    wn = wn_vec[i-1]

                # Calculate soil moisture and runoff:
                sm, ro = self.quick_run(n=i+1,
                                        y=d.year,
                                        wn=wn,
                                        sf=d.sf_vec[i],
                                        tc=d.tair_vec[i],
                                        pn=d.pn_vec[i])
                wn_vec[i] = sm

            start_sm = wn_vec[0]
            end_sm, ro = self.quick_run(n=1,
                                        y=d.year,
                                        wn=wn_vec[-1],
                                        sf=d.sf_vec[0],
                                        tc=d.tair_vec[0],
                                        pn=d.pn_vec[0])
            diff_sm = numpy.abs(end_sm - start_sm)
            self.logger.info("soil moisture differential: %f", diff_sm)
            spin_count += 1
        self.logger.info("equilibrated after %d iterations", spin_count)
        self.wn_vec = wn_vec
        self.wn = wn_vec[-1]

    def quick_run(self, n, y, wn, sf, tc, pn):
        """
        Name:     SPLASH.quick_run
        Inputs:   - int, day of year (n)
                  - int, year (y)
                  - float, daily soil water content, mm (wn)
                  - float, daily fraction of bright sunshine (sf)
                  - float, daily air temperature, deg C (tc)
                  - float, daily precipitation, mm (pn)
        Output:   - float, soil moisture, mm (sm)
                  - float, runoff, mm (ro)
        Features: Returns daily soil moisture and runoff.
        Depends:  - kCw
                  - kWm
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Calculate evaporative supply rate, mm/h
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sw = kCw*(wn/kWm)
        self.logger.debug("evaporative supply rate: %f mm/h", sw)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Calculate radiation and evaporation quantities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        try:
            self.evap.calculate_daily_fluxes(sw, n, y, sf, tc)
        except:
            self.logger.exception(
                "failed to calculate daily evaporation fluxes")
            raise
        else:
            cond = self.evap.cond
            aet = self.evap.aet_d

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate today's soil moisture, mm
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sm = wn + pn + cond - aet
        self.logger.debug("calculated soil moisture as %f mm", sm)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Calculate runoff, mm
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if sm > kWm:
            self.logger.debug("bucket is too full")
            # Bucket is too full
            #   allocate excess water to runoff
            #   set soil moisture to capacity (i.e., kWm)
            ro = sm - kWm
            sm = kWm
            self.logger.debug("soil moisture: %f mm", sm)
            self.logger.debug("excess runoff: %f mm", ro)
        elif sm < 0:
            self.logger.debug("bucket is too empty")
            # Bucket is too empty
            #   set soil moisture and runoff to zero
            sm = 0
            ro = 0
            self.logger.debug("soil moisture: %d mm", sm)
            self.logger.debug("excess runoff: %d mm", ro)
        else:
            ro = 0
            self.logger.debug("excess runoff: %d mm", ro)

        return(sm, ro)

    def run_one_day(self, n, y, wn, sf, tc, pn):
        """
        Name:     SPLASH.run_one_day
        Inputs:   - int, day of year (n)
                  - int, year (y)
                  - float, soil water content, mm (wn)
                  - float, fraction of bright sunshine (sf)
                  - float, air temperature, deg C (tc)
                  - float, precipitation, mm (pn)
        Outputs:  None
        Features: Runs SPLASH model for one day.
        Depends:  - kCw
                  - kWm
                  - EVAP
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 0. Set meteorological variables:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.precip = pn    # daily precipitation, mm
        self.logger.debug("daily precipitation: %f mm", pn)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Calculate evaporative supply rate (sw), mm/h
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sw = kCw*float(wn)/kWm
        self.logger.debug("evaporative supply rate: %f mm/h", sw)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Calculate radiation and evaporation quantities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        try:
            self.evap.calculate_daily_fluxes(sw, n, y, sf, tc)
        except:
            self.logger.exception(
                "failed to calculate daily evaporation fluxes")
            raise
        else:
            self.cond = self.evap.cond    # daily condensation, mm
            self.aet = self.evap.aet_d    # daily actual ET, mm
            self.eet = self.evap.eet_d    # daily equilibrium ET, mm
            self.pet = self.evap.pet_d    # daily potential ET, mm

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate today's soil moisture (sm), mm
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sm = wn + pn + self.cond - self.aet
        self.logger.debug("calculated soil moisture as %f mm", sm)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Calculate runoff (ro), mm
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if sm > kWm:
            self.logger.debug("bucket is too full")
            self.logger.debug("setting soil moisture to saturation")
            self.logger.debug("calculating runoff")
            # Bucket is too full
            #   allocate excess water to runoff
            #   set soil moisture to capacity (i.e., kWm)
            ro = sm - kWm
            sm = kWm
        elif sm < 0:
            self.logger.debug("bucket is too empty")
            self.logger.debug("correcting actual ET")
            # Bucket is too empty
            #   reduce actual ET by discrepancy amount
            #   set soil moisture and runoff to zero
            self.aet += sm
            sm = 0
            ro = 0
        else:
            ro = 0
        self.logger.debug("soil moisture: %f mm", sm)
        self.logger.debug("excess runoff: %f mm", ro)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. Update soil moisture & runoff
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.wn = sm  # daily soil moisture, mm
        self.ro = ro  # daily runoff, mm

    def print_vals(self):
        """
        Name:     SPLASH.print_vals
        Inputs:   None
        Outputs:  None
        Features: Prints all daily values
        """
        print("Daily values:")
        print("  Pn: %0.6f mm" % (self.precip))
        print("  Cn: %0.6f mm" % (self.cond))
        print("  EET: %0.6f mm" % (self.eet))
        print("  PET: %0.6f mm" % (self.pet))
        print("  AET: %0.6f mm" % (self.aet))
        print("  Wn: %0.6f mm" % (self.wn))
        print("  RO: %0.6f mm" % (self.ro))

    def print_daily_sm(self):
        """
        Name:     SPLASH.print_daily_sm
        Inputs:   None.
        Outputs:  None.
        Features: Prints the daily soil moisture values
        """
        print("Day,Wn (mm)")
        for i in range(len(self.wn_vec)):
            print("%d,%0.6f" % (i, self.wn_vec[i]))

###############################################################################
# MAIN PROGRAM
###############################################################################
if __name__ == '__main__':
    # Create a root logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Instantiating logging handler and record format:
    root_handler = logging.FileHandler("splash.log")
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
    my_sm = 75
    my_precip = 5

    my_class = SPLASH(my_lat, my_elv)
    my_class.run_one_day(my_day, my_year, my_sm, my_sf, my_temp, my_precip)
    my_class.print_vals()
