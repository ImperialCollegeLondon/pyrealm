#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# data.py
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
## IMPORT MODULES:
###############################################################################
import logging

import numpy


###############################################################################
## CLASSES
###############################################################################
class DATA:
    """
    Name:     DATA
    Features: This class handles the file IO for reading and writing data.
    History:  Version 1.0
              - added logging statements [16.02.05]
    """
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Initialization
    # ////////////////////////////////////////////////////////////////////////
    def __init__(self):
        """
        Name:     DATA.__init__
        Input:    str, input file name (fname)
        Features: Initialize empty class variables
        """
        # Create a class logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("DATA class called")

        self.file_name = ""
        self.year = 0
        self.num_lines = 0.
        self.sf_vec = numpy.array([])
        self.tair_vec = numpy.array([])
        self.pn_vec = numpy.array([])

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Function Definitions
    # ////////////////////////////////////////////////////////////////////////
    def read_csv(self, fname, y=-1):
        """
        Name:     DATA.read_csv
        Input:    - str, input CSV filename (fname)
                  - int, year (y)
        Output:   None
        Features: Reads all three daily input variables (sf, tair, and pn) for
                  a single year from a CSV file that includes a headerline.
        """
        self.file_name = fname

        try:
            data = numpy.loadtxt(fname,
                                 dtype={'names': ('sf', 'tair', 'pn'),
                                        'formats': ('f4', 'f4', 'f4')},
                                 delimiter=',',
                                 skiprows=1)
        except IOError:
            self.logger.exception("could not read input file %s", fname)
            raise
        else:
            self.sf_vec = data['sf']
            self.tair_vec = data['tair']
            self.pn_vec = data['pn']
            self.num_lines = data.shape[0]

            if y == -1:
                if data.shape[0] == 366:
                    self.year = 2000
                elif data.shape[0] == 365:
                    self.year = 2001
            else:
                self.year = y

    def read_txt(self, fname, var, y=-1):
        """
        Name:     DATA.read_txt
        Input:    - str, input text file (fname)
                  - str, variable name (i.e., 'pn', 'sf', 'tair')
                  - int, year (y)
        Output:   None.
        Features: Reads plain text file (no header) into one of daily input
                  arrays.
        """
        # Add filename to list:
        if not isinstance(self.file_name, list):
            self.file_name = []
        self.file_name.append(fname)

        try:
            data = numpy.loadtxt(fname, dtype='f4')
        except IOError:
            self.logger.exception("could not read input file %s", fname)
            raise
        else:
            if var == 'sf':
                self.sf_vec = data
            elif var == 'pn':
                self.pn_vec = data
            elif var == 'tair':
                self.tair_vec = data
            else:
                self.logger.error("variable %s undefined!", var)
                raise ValueError("Unrecognized variable in read_txt")

            # Add line numbers to list:
            if not isinstance(self.num_lines, list):
                self.num_lines = []
            self.num_lines.append(data.shape[0])

            if y == -1:
                if data.shape[0] == 366:
                    self.year = 2000
                elif data.shape[0] == 365:
                    self.year = 2001
            else:
                self.year = y
