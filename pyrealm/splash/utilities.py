"""Utilities for the SPLASH module."""
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

    # TODO - could be replaced with xarray dt accessors?
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

    def __iter__(self) -> Generator[CalendarDay, Any, Any]:
        """Yield each date in the Calendar in sequence."""
        for idx, dt in enumerate(self.dates):
            yield CalendarDay(
                date=dt,
                year=self.year[idx],
                julian_day=self.julian_day[idx],
                days_in_year=self.days_in_year[idx],
            )

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
        return len(self.dates)
