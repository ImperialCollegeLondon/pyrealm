"""Core shared functionality for the {mod}`~pyrealm.demography` module."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class PandasExporter(ABC):
    """Abstract base class implementing pandas export.

    Classes inheriting from this ABC must define a class attribute ``array_attrs`` that
    names a set of class attributes that are all numpy arrays of equal length. The
    classes then inherit the `to_pandas` method that exports those attributes to a
    {class}`pandas.DataFrame`.
    """

    array_attrs: ClassVar[tuple[str, ...]]

    def to_pandas(self) -> pd.DataFrame:
        """Convert the instance array attributes into a {class}`pandas.DataFrame."""
        return pd.DataFrame({k: getattr(self, k) for k in self.array_attrs})


class Cohorts(ABC):
    """Abstract base class implementing cohort manipulation functionality.

    Classes inheriting from this ABC must define a class attribute ``array_attrs`` that
    names a set of class attributes that are all numpy arrays of equal length. The class
    then inherit:

    * The `add_cohorts` method, which allows a second instance of the same class to be
      joined to the calling instance, concatenting each of the array attributes from
      the second instance onto the calling instance.
    * The `drop_cohorts` method, which takes a set of indices onto the array attributes
      and drops the values from those indices for each array attribute.
    """

    array_attrs: ClassVar[tuple[str, ...]]

    def add_cohorts(self, add: Cohorts) -> None:
        """Add array attributes from a second instance.

        Args:
            add: A second instance from which to add array attribute values.
        """
        for trait in self.array_attrs:
            setattr(
                self,
                trait,
                np.concatenate([getattr(self, trait), getattr(add, trait)]),
            )

    def drop_cohorts(self, drop_indices: NDArray[np.int_]) -> None:
        """Drop array attribute values from an instance.

        Args:
            drop_indices: An array of integer indices to drop from each array attribute.
        """

        for trait in self.array_attrs:
            setattr(self, trait, np.delete(getattr(self, trait), drop_indices))
