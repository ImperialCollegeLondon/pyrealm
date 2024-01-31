"""Provides the Calendar utility class.

The Calendar class is currently used to support the operation of the
:mod:`~pyrealm.splash` submodule, which requires the Julian day, number of days in the
year and year for solar calculations. The class provides iterable and indexable access
to a date sequence for use in those calculations.

It is possible that this could be replaced with xarray dt accessors if pyrealm adopts
xarray data structures.
"""
from dataclasses import dataclass, field
from typing import Any, Generator, Sized

import numpy as np


@dataclass
class CalendarDay:
    """The CalendarDay class.

    This dataclass holds a np.datetime64 datetime representing a day and the
    corresponding year, julian day and days in year.
    """

    date: np.datetime64
    year: int
    julian_day: int
    days_in_year: int


@dataclass
class Calendar(Sized):
    """The Calendar class.

    This utility class takes a numpy array of datetime64 values containing a time series
    of individual days and calculates the date, year, julian day and days in the year
    for each observation. The class object can be iterated over as a generator, yielding
    each date in turn and indexed. In both cases, the returned object is a CalendarDay
    instance.
    """

    dates: np.ndarray
    year: np.ndarray = field(init=False)
    julian_day: np.ndarray = field(init=False)
    days_in_year: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Calculate year, julian day and days in year from dates."""
        dateyear = self.dates.astype("datetime64[Y]")
        startnext = (dateyear + 1).astype("datetime64[D]")
        dateday = self.dates.astype("datetime64[D]")
        self.year = dateyear.astype("int") + 1970
        self.julian_day = (dateday - dateyear + 1).astype("int")
        self.days_in_year = (startnext - dateyear).astype("int")
        self.n_dates = len(self.dates)

    def __iter__(self) -> Generator[CalendarDay, Any, Any]:
        """Yield each date in the Calendar in sequence."""
        for idx in range(self.n_dates):
            yield self[idx]

    def __getitem__(self, idx: int) -> CalendarDay:
        """Extract dates by index."""
        return CalendarDay(
            date=self.dates[idx],
            year=self.year[idx],
            julian_day=self.julian_day[idx],
            days_in_year=self.days_in_year[idx],
        )

    def __len__(self) -> int:
        """Length of a Calendar object."""
        return self.n_dates
