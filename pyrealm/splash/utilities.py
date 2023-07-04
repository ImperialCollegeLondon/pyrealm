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
