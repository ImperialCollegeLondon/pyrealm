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

    dates: np.ndarray
    """A numpy array containing :class:`numpy.datetime64` values."""
    year: np.ndarray = field(init=False)
    """A numpy array giving the year of each datetime."""
    julian_day: np.ndarray = field(init=False)
    """A numpy array giving the julian day of each datetime."""
    days_in_year: np.ndarray = field(init=False)
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


@dataclass
class LocationDateTime:
    """A data class representing an observation location and date and time information.

    This class encapsulates the latitude and longitude of a location along with a
    corresponding time array. It automatically calculates the latitude and longitude in
    radians, the Julian days from the date-time information, and a decimal
    representation of the local time.

    Attributes:
        latitude        : The latitude of the location in degrees.
        latitude_rad    : The latitude of the location in radians, calculated
                            automatically.
        longitude       : The longitude of the location in degrees.
        longitude_rad   : The longitude of the location in radians, calculated
                            automatically.
        UTC_offset      : The offset from Coordinated Universal Time (UTC) for the
                            location.
        year_date_time  : An array of bp.datetime64 values corresponding to observations
                            at the location.
        julian_days     : An array of Julian day of the year numbers calculated
                            from the `year_date_time`.
        local_time      : An array of local times in decimal hour format
                            calculated from the `year_date_time`.

    Methods:
        __post_init__() : Initializes calculated fields like `latitude_rad`,
                            `longitude_rad`, `julian_days`, and `local_time` after the
                            object is instantiated.
        decimal_hour()  : Converts the `year_date_time` to a decimal
                            representation of hours.

    Example:
        >>> import numpy as np
        >>> ldt = LocationDateTime(
        ...     latitude=-35.058333,
        ...     longitude=147.34167,
        ...     UTC_offset=10,
        ...     year_date_time=np.array([np.datetime64("2024-08-12T10:30")])
        ... )
        >>> print(ldt.latitude_rad)
        -0.6118833411105811
        >>> print(ldt.local_time)
        [10.5]
    """

    latitude: float
    latitude_rad: float = field(init=False)
    longitude: float
    longitude_rad: float = field(init=False)
    UTC_offset: int
    year_date_time: np.ndarray
    julian_days: np.ndarray = field(init=False)
    local_time: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.julian_days = Calendar(self.year_date_time).julian_day
        self.local_time = self.decimal_hour()
        self.latitude_rad = self.latitude * np.pi / 180
        self.longitude_rad = self.longitude * np.pi / 180

    def decimal_hour(self) -> np.ndarray:
        """Convert year_date_time to a decimal representation of hours.

        This method extracts the hours and minutes from the `year_date_time` attribute
        and converts them into a decimal representation of hours.

        Returns:
            An array of decimal hour values.
        """

        # Extract hours
        hours = self.year_date_time.astype("datetime64[h]").astype(int) % 24

        # Extract minutes
        minutes = self.year_date_time.astype("datetime64[m]").astype(int) % 60

        # Convert to decimal hours
        return hours + minutes / 60
