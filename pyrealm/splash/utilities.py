#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# utilities.py
#
# VERSION: 1.1-dev
# LAST UPDATED: 2017-04-28
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
import glob
import logging
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from pyrealm.splash.const import pir


@dataclass
class Calendar:
    date: np.ndarray[np.datetime64]
    year: np.ndarray[int] = field(init=False)
    julian_day: np.ndarray[int] = field(init=False)
    days_in_year: np.ndarray[int] = field(init=False)

    def __post_init__(self):
        dateyear = self.date.astype("datetime64[Y]")
        startnext = (dateyear + 1).astype("datetime64[D]")
        dateday = self.date.astype("datetime64[D]")
        self.year = dateyear.astype("int") + 1970
        self.julian_day = (dateday - dateyear + 1).astype("int")
        self.days_in_year = (startnext - dateyear).astype("int")

    def __iter__(self):
        for idx, dt in enumerate(self.date):
            yield CalendarDay(
                date=dt,
                year=self.year[idx],
                julian_day=self.julian_day[idx],
                days_in_year=self.days_in_year[idx],
            )

    def __getitem__(self, idx: int):
        return CalendarDay(
            date=self.date[idx],
            year=self.year[idx],
            julian_day=self.julian_day[idx],
            days_in_year=self.days_in_year[idx],
        )


@dataclass
class CalendarDay:
    date: np.datetime64
    year: int
    julian_day: int
    days_in_year: int


###############################################################################
# FUNCTIONS:
###############################################################################
def calculate_latitude(y, r):
    """
    Name:     calculate_latitude
    Input:    - nd.array, latitude index (y)
              - float, pixel resolution (r)
    Output:   nd.array, latitude, degrees
    Features: Returns latitude for an index array (numbered from the
              bottom-left corner) and pixel resolution
    """
    # Offset lat to pixel centroid and calucate based on index:
    lat = -90.0 + (0.5 * r)
    lat += y * r
    if isinstance(y, int):
        logging.debug("latitude at index %d is %f at resolution %0.3f" % (y, lat, r))
    else:
        logging.debug("calculating latitude at %0.3f degrees", r)
    return lat


def dcos(x):
    """
    Name:     dcos
    Input:    float/nd.array, angle, degrees (x)
    Output:   float/nd.array, cos(x*pi/180)
    Features: Calculates the cosine of an array of angles in degrees
    """
    if isinstance(x, float):
        logging.debug("calculating cosine of %f degrees", x)
    elif isinstance(x, np.ndarray):
        logging.debug("calculating cosine of numpy array of length %d", x.size)
    return np.cos(x * pir)


def dsin(x):
    """
    Name:     dsin
    Input:    float/nd.array, angle, degrees (x)
    Output:   float/nd.array, sin(x*pi/180)
    Features: Calculates the sine of an array of angles in degrees
    """
    if isinstance(x, float):
        logging.debug("calculating sine of %f degrees", x)
    elif isinstance(x, np.ndarray):
        logging.debug("calculating cosine of numpy array of length %d", x.size)
    return np.sin(x * pir)


def find_files(my_dir, my_pattern):
    """
    Name:     find_files
    Inputs:   - str, directory path (my_dir)
              - str, file name search pattern (my_pattern)
    Outputs:  list, file paths
    Features: Returns a sorted list of files found at a given directory with
              file names that match a given pattern
    """
    my_files = []
    if os.path.isdir(my_dir):
        s_str = os.path.join(my_dir, my_pattern)
        my_files = glob.glob(s_str)
    if len(my_files) > 0:
        my_files = sorted(my_files)
    return my_files


def get_x_y(lon, lat, r=0.5):
    """
    Name:     get_x_y
    Input:    - float, longitude, degrees (lon)
              - float, latitude, degrees (lat)
              - [optional] float, resolution, degrees (r)
    Output:   tuple, x-y indices
    Features: Returns x and y indices for a given lon-lat pair and pixel
              resolution. Y is latitude (0--359). X is longitude (0--719)
    """
    if lon > 179.75 or lon < -179.75:
        logging.warning("longitude outside range of validity")
        return (None, None)
    elif lat > 89.75 or lat < -89.75:
        logging.warning("latitude outside range of validity")
        return (None, None)
    else:
        # Solve x and y indices:
        x = (lon + 180.0) / r - 0.5
        y = (lat + 90.0) / r - 0.5
        return (int(x), int(y))


def print_progress(iteration, total, prefix="", suffix="", decimals=0, bar_length=44):
    """
    Name:     print_progress
    Inputs:   - int, current iteration (iteration)
              - int, total iterations (total)
              - [optional] str, prefix string (prefix)
              - [optional] str, suffix string (suffix)
              - [optional] int, decimal number in percent complete (decimals)
              - [optional] int, character length of bar (bar_length)
    Outputs:  None.
    Features: Creates a terminal progress bar
    Reference: "Python Progress Bar" by aubricus
    https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),

    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()
