"""Some functions in ``pyrealm`` are only well-behaved with given bounds but those
bounds are often a little imprecise and real world data can contain extreme values. As a
result, the bounds checking is deliberately not that intrusive: it warns when a variable
contains out of value issues but leaves it up to the user to assess whether there is
real problem and to adjust input data if needed.

The ``bounds`` module:

* Defines a {class}`~pyrealm.core.bounds.Bounds` dataclass used to define bounds for a
  particular variable.
* Defines a {class}`~pyrealm.core.bounds.BoundsChecker` class with default bounds for
  core variables that acts as a library for bounds checking.
* The main use case is e.g. ``BoundsChecker().check("tc", np.array([10, 1000])``, which
  will check that the alleged temperature data in °C fall within the configured bounds.

A ``BoundsChecker`` class instance is created with a predefined internal dictionary of
default variables and appropriate bounds. However, users can use the
{meth}`~pyrealm.core.bounds.BoundsChecker.update` method to overide defaults or add new
variables by providing a new ``Bounds`` instance.

The {meth}`~pyrealm.core.bounds.BoundsChecker.check` method can then be used to validate
a set of values against the configured bounds for a given variable name. The ``check``
method returns the input variables, to allow values to be checked while being assigned
to an attribute.
"""  # noqa: D205

from dataclasses import dataclass
from typing import Any, ClassVar
from warnings import warn

import numpy as np
from numpy.typing import NDArray


@dataclass
class Bounds:
    """Bounds checking dataclass for variables."""

    var_name: str
    """A variable name, typically the form used in function arguments."""
    lower: float
    """A lower bound on sensible values."""
    upper: float
    """An upper bound on sensible values."""
    interval_type: str
    """The interval type of the constraint ('[]', '()', '[)', '(]')."""
    unit: str
    """A string giving the expected units."""

    def __post_init__(self) -> None:
        """Bounds data validation."""
        if self.interval_type not in BoundsChecker._interval_types:
            raise ValueError(f"Unknown interval type: {self.interval_type}")

        if self.lower >= self.upper:
            raise ValueError(f"Bounds equal or reversed: {self.lower}, {self.upper}")


class BoundsChecker:
    """A bounds checker for input variables.

    The class provides a library of  {class}`~pyrealm.core.bounds.Bounds` instances for
    core variables, keyed by the
    {attr}`Bounds.var_name<pyrealm.core.bounds.Bounds.var_name>` attribute. The table is
    populated from default values when a ``BoundsChecker`` instance is created but can
    be updated and extended by assigning new ``Bounds`` instances to existing or new
    variable name keys using the ``update`` method.
    """

    # TODO - think about these argument names - some unnecessarily terse.
    _defaults: tuple[tuple[str, float, float, str, str], ...] = (
        ("tc", -25, 80, "[]", "°C"),
        ("vpd", 0, 10000, "[]", "Pa"),
        ("co2", 0, 1000, "[]", "ppm"),
        ("patm", 30000, 110000, "[]", "Pa"),
        ("fapar", 0, 1, "[]", "-"),
        ("ppfd", 0, 3000, "[]", "µmol m-2 s-1"),
        ("theta", 0, 0.8, "[]", "m3 m-3"),
        ("rootzonestress", 0, 1, "[]", "-"),
        ("aridity_index", 0, 50, "[]", "-"),
        ("mean_growth_temperature", 0, 50, "[]", "-"),
        ("rh", 0, 1, "[]", "-"),
        ("lat", -90, 90, "[]", "°"),
        ("sf", 0, 1, "[]", "-"),
        ("pn", 0, 1000, "[]", "mm day-1"),
        ("kWm", 0, 1e4, "[]", "mm"),
        ("leaf_area_index", 0, 20, "[]", "-"),
        ("solar_elevation", -90, 90, "[]", "degrees"),
    )
    """Default bounds data for core forcing variables."""

    _interval_types: ClassVar[dict[str, tuple[np.ufunc, np.ufunc]]] = {
        "()": (np.greater, np.less),
        "[]": (np.greater_equal, np.less_equal),
        "(]": (np.greater, np.less_equal),
        "[)": (np.greater_equal, np.less),
    }
    """Dictionary of numpy function pairs for testing interval types."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._data: dict[str, Bounds] = {}

        for var in self._defaults:
            var_bounds = Bounds(*var)
            self._data[var_bounds.var_name] = var_bounds

    def update(self, bounds: Bounds) -> None:
        """Update or add bounds data.

        The {attr}`Bounds.var_name<pyrealm.core.bounds.Bounds.var_name>` attribute of
        the provided ``Bounds`` instance is used to update an existing entry for the
        name or add checking for a new name.

        Args:
            bounds: A Bounds instance.
        """

        self._data[bounds.var_name] = bounds

    def check(self, var_name: str, values: NDArray) -> NDArray:
        r"""Check inputs fall within bounds.

        This method checks whether the provided values fall within the bounds specified
        for the given variable name and issues a warning when this is not the case. If
        the ``BoundsChecker`` class has not been configured the variable name then a
        warning will be given about lack of bounds checking. The method returns the
        input values, so that the method can be used as a pass through validator for
        assigning attributes.

        Args:
            var_name: The variable name
            values: An np.ndarray

        Returns:
            The input values.

        Examples:
            >>> vals = np.array([-15, 20, 30, 124], dtype=float)
            >>> bounds_checker = BoundsChecker()
            >>> bounds_checker.check("temp", vals)
            array([-15.,  20.,  30., 124.])
        """

        var_bounds = self._data.get(var_name)

        if var_bounds is None:
            warn(
                f"Variable '{var_name}' is not configured in the bounds checker. "
                "No bounds checking performed."
            )
            return values

        # Get the interval functions
        lower_func, upper_func = self._interval_types[var_bounds.interval_type]

        # Do the input values contain out of bound values?
        out_of_bounds = np.logical_xor(
            lower_func(values, var_bounds.lower),
            upper_func(values, var_bounds.upper),
        )

        if np.any(out_of_bounds):
            warn(
                f"Variable '{var_name}' ({var_bounds.unit}) contains values outside "
                f"the expected range ({var_bounds.lower},{var_bounds.upper}). "
                "Check units?"
            )

        return values
