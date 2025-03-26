"""Provides the Calendar and CalendarDay utility classes.

The :class:`pyrealm.core.calendar.Calendar` class is currently used to support the
operation of the :mod:`~pyrealm.splash` submodule, which requires the Julian day, number
of days in the year and year for solar calculations. The class provides iterable and
indexable access to a date sequence for use in those calculations, returning individual
:class:`pyrealm.core.calendar.CalendarDay` instances.

It is possible that this could be replaced with xarray dt accessors if pyrealm adopts
xarray data structures.
"""

from collections.abc import Generator, Sized
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CalendarDay:
    """The CalendarDay class.

    This dataclass holds a :class:`numpy.datetime64` datetime representing a day and the
    corresponding year, julian day (the integer index of the day within the year)
    and the total number of days in the year.
    """

    date: np.datetime64
    """The date of the instance as numpy.datetime64."""
    year: int
    """The year of the instance as an integer."""
    julian_day: int
    """The julian day of the instance as an integer."""
    days_in_year: int
    """The total number of days in the year."""

    def __repr__(self) -> str:
        """A custom representation method.

        The dataclass implementation does not support custom field representation, so
        this replaces the default.
        """
        return (
            f"CalendarDay(date={self.date!s}, year={self.year}, "
            f"julian_day={self.julian_day}, days_in_year={self.days_in_year})"
        )


@dataclass
class Calendar(Sized):
    """The Calendar class.

    This utility class takes a numpy array of :class:`numpy.datetime64` values
    representing a time series of individual days and calculates the date, year, julian
    day and days in the year for each observation. The class object can be iterated
    over, yielding each date in turn, and dates within the series can also be accessed
    by index. In both cases, the returned object is a
    :class:`~pyrealm.core.calendar.CalendarDay` instance.

    Examples:
        >>> days=np.arange(
        ...     np.datetime64("2000-01-01"),
        ...     np.datetime64("2002-01-01"),
        ...     np.timedelta64(1, "D")
        ... )
        >>> cal = Calendar(days)
        >>> cal
        Calendar(2000-01-01, 2001-12-31)
        >>> len(cal) == (366 + 365)
        True
        >>> cal[0]
        CalendarDay(date=2000-01-01, year=2000, julian_day=1, days_in_year=366)
        >>> for date in cal:
        ...     pass
        >>> date
        CalendarDay(date=2001-12-31, year=2001, julian_day=365, days_in_year=365)
    """

    dates: NDArray[np.datetime64]
    """A numpy array containing :class:`numpy.datetime64` values."""
    year: NDArray[np.int_] = field(init=False)
    """A numpy array giving the year of each datetime."""
    julian_day: NDArray[np.int_] = field(init=False)
    """A numpy array giving the julian day of each datetime."""
    days_in_year: NDArray[np.int_] = field(init=False)
    """A numpy array giving the number of days in the year for each datetime."""

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

    def __repr__(self) -> str:
        """Representation of a Calendar instance."""

        return f"Calendar({self.dates[0]!s}, {self.dates[-1]!s})"
