#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# main.py
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
#
# ~~~~~~~~~~
# changelog:
# ~~~~~~~~~~
# 00. created script based on cramer_prentice.py [14.01.30]
# 01. global constants [14.08.26]
# 02. EVAP class [14.08.26]
# 03. moved class constants to global constants
# 04. updated komega to float --- may influence cooper delta [14.09.29]
# 05. added 'berger' lamm method [14.09.30]
# 06. added check to woolf's method for lambda > 360 [14.10.01]
# 07. added Spencer method for declination [14.10.10]
# 08. replaced tau with Allen (1996); removed kZ [14.10.10]
# 09. distinguished shortwave from visible light albedo [14.10.16]
# 10. updated value and reference for semi-major axis, a [14.10.31]
# 11. fixed Cooper's and Spencer's declination equations [14.11.25]
# 12. replaced simplified kepler with full kepler [14.11.25]
# 13. removed options for approximation methods not considering variable
#     orbital velocity (e.g. Spencer, Woolf, Klein, Cooper, and Circle
#     methods) [14.12.09]
# 14. reduced the list of constants and EVAP class functions [14.12.09]
# 15. added matplotlib to module list [14.12.09]
# 16. added plots for results [14.12.09]
# 17. removed longitude from EVAP & STASH classes [15.01.13]
# 18. general housekeeping [15.01.13]
# 19. updated plots for results [15.01.16]
# 20. added example data CSV file & updated data for daily input [15.01.16]
# 21. fixed spin_up indexing in STASH class [15.01.16]
# 22. fixed Cramer-Prentice alpha definition [15.01.16]
# 23. updated plots [15.01.18]
# 24. updated reference to kL [15.01.29]
# 25. general housekeeping on EVAP class [15.02.07]
# 25. changed condensation variable name from 'wc' to 'cn' [15.02.07]
# 26. created DATA class for file IO handling [15.02.09]
#     --> read all data from single CSV file
#     --> OR read each variable from individual text files
# 27. updated STASH class to run for one day [15.02.09]
#     --> spin-up function still creates a soil moisture array
# 28. updated R and To values and references [15.08.22]
# 29. parsed classes into separate python files [15.08.22]
# 30. created a global constant file, const.py [15.08.22]
# 31. added logging [16.02.05]
#
###############################################################################
## IMPORT MODULES
###############################################################################
import logging

from data import DATA
from splash import SPLASH

###############################################################################
## MAIN PROGRAM
###############################################################################
if __name__ == '__main__':
    # Create a root logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Instantiating logging handler and record format:
    root_handler = logging.FileHandler("main.log")
    rec_format = "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s"
    formatter = logging.Formatter(rec_format, datefmt="%Y-%m-%d %H:%M:%S")
    root_handler.setFormatter(formatter)

    # Send logging handler to root logger:
    root_logger.addHandler(root_handler)

    example = 1
    my_data = DATA()
    if example == 1:
        # Example 1: read CSV file:
        my_file = '../../../data/example_data.csv'
        my_data.read_csv(my_file)
    elif example == 2:
        # Example 2: read TXT files:
        my_sf_file = 'daily_sf_2000_cruts.txt'
        my_pn_file = 'daily_pn_2000_wfdei.txt'
        my_tair_file = 'daily_tair_2000_wfdei.txt'
        my_data.read_txt(my_sf_file, 'sf')
        my_data.read_txt(my_pn_file, 'pn')
        my_data.read_txt(my_tair_file, 'tair')

    # Consistency Test #4: Spin-Up
    my_lat = 37.7
    my_elv = 142.
    my_class = SPLASH(my_lat, my_elv)
    my_class.spin_up(my_data)
    my_class.print_daily_sm()
