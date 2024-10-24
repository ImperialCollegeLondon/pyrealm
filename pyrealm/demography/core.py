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
        """Convert the instance array attributes into a {class}`pandas.DataFrame.

        If the array values are two-dimensional (i.e. stem or cohort data by vertical
        heights), the data are stacked and an index is added.
        """

        # Extract the attributes into a dictionary
        data = {k: getattr(self, k) for k in self.array_attrs}

        # Check the first attribute array to see if the values are two dimensional
        data_shape = data[self.array_attrs[0]].shape

        if len(data_shape) == 2:
            # create an index entry to show the column of each value
            stacked_data = {
                "column_stem_index": np.repeat(np.arange(data_shape[1]), data_shape[0])
            }
            # Ravel the attribute data using column-major Fortan style
            for ky, vl in data.items():
                stacked_data[ky] = np.ravel(vl, order="F")

            return pd.DataFrame(stacked_data).set_index("column_stem_index")

        return pd.DataFrame(data)


class CohortMethods(ABC):
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

    def add_cohort_data(self, new_data: CohortMethods) -> None:
        """Add array attributes from a second instance implementing the base class.

        Args:
            new_data: A second instance from which to add cohort data to array attribute
                values.
        """

        # Check the incoming dataclass matches the calling instance.
        if not isinstance(new_data, self.__class__):
            raise ValueError(
                f"Cannot add cohort data from an {type(new_data).__name__} "
                f"instance to {type(self).__name__}"
            )

        # Concatenate the array attributes from the incoming instance to the calling
        # instance.
        for trait in self.array_attrs:
            setattr(
                self,
                trait,
                np.concatenate([getattr(self, trait), getattr(new_data, trait)]),
            )

    def drop_cohort_data(self, drop_indices: NDArray[np.int_]) -> None:
        """Drop array attribute values from an instance.

        Args:
            drop_indices: An array of integer indices to drop from each array attribute.
        """

        # TODO - Probably part of tackling #317
        #        The delete axis=0 here is tied to the case of dropping rows from 2D
        #        arrays, but then I'm thinking it makes more sense to _only_ support 2D
        #        arrays rather than the current mixed bag of getting a 1D array when a
        #        single height is provided. Promoting that kind of input to 2D and then
        #        enforcing an identical internal structure seems better.
        #      - But! Trait data does not have 2 dimensions!
        #      - Also to check here - this can lead to empty instances, which probably
        #        are a thing we want, if mortality removes all cohorts.

        for trait in self.array_attrs:
            setattr(self, trait, np.delete(getattr(self, trait), drop_indices, axis=0))
