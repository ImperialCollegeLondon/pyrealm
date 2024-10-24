"""Core shared functionality for the {mod}`~pyrealm.demography` module.

This module implements two abstract base classes that are used to share core methods
across demography classes:

* {class}`~pyrealm.demography.core.PandasExporter` provides the utility
  {meth}`~pyrealm.demography.core.PandasExporter.to_pandas` method for extracting data
  from demography classes for plotting and exploring data.
* {class}`~pyrealm.demography.core.CohortMethods` provides the utility
  {meth}`~pyrealm.demography.core.CohortMethods.add_cohort_data` and
  {meth}`~pyrealm.demography.core.CohortMethods.drop_cohort_data` methods that are used
  to append new cohort data across some demography dataclasses.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyrealm.core.utilities import check_input_shapes


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


def _validate_z_qz_args(
    z: NDArray[np.float64],
    stem_properties: list[NDArray[np.float64]],
    q_z: NDArray[np.float64] | None = None,
) -> None:
    """Shared validation of for crown function arguments.

    Several of the crown functions in this module require a vertical height (``z``)
    argument and, in some cases, the relative crown radius (``q_z``) at that height.
    These arguments need to have shapes that are congruent with each other and with the
    arrays providing stem properties for which values are calculated.

    This function provides the following validation checks (see also the documentation
    of accepted shapes for ``z`` in
    :meth:`~pyrealm.demography.crown.calculate_relative_crown_radius_at_z`).

    * Stem properties are identically shaped row (1D) arrays.
    * The ``z`` argument is then one of:
        * a scalar arrays (i.e. np.array(42) or np.array([42])),
        * a row array with identical shape to the stem properties, or
        * a column vector array (i.e. with shape ``(N, 1)``).
    * If ``q_z`` is provided then:
        * if ``z`` is a row array, ``q_z`` must then have identical shape, or
        * if ``z`` is a column array ``(N, 1)``, ``q_z`` must then have shape ``(N,
          n_stem_properties``).

    Args:
        z: An array input to the ``z`` argument of a crown function.
        stem_properties: A list of array inputs representing stem properties.
        q_z: An optional array input to the ``q_z`` argument of a crown function.
    """

    # Check the stem properties
    try:
        stem_shape = check_input_shapes(*stem_properties)
    except ValueError:
        raise ValueError("Stem properties are not of equal size")

    if len(stem_shape) > 1:
        raise ValueError("Stem properties are not row arrays")

    # Record the number of stems
    n_stems = stem_shape[0]

    # Trap error conditions for z array
    if z.size == 1:
        pass
    elif (z.ndim == 1) and (z.shape != stem_shape):
        raise ValueError(
            f"The z argument is a row array (shape: {z.shape}) but is not congruent "
            f"with the cohort data (shape: {stem_shape})."
        )
    elif (z.ndim == 2) and (z.shape[1] != 1):
        raise ValueError(
            f"The z argument is two dimensional (shape: {z.shape}) but is "
            "not a column array."
        )
    elif z.ndim > 2:
        raise ValueError(
            f"The z argument (shape: {z.shape}) is not a row or column vector array"
        )

    # Now test q_z congruence with z if provided
    if q_z is not None:
        if q_z.shape == z.shape:
            pass
        elif ((z.size == 1) or (z.ndim == 1)) and (q_z.shape != stem_shape):
            raise ValueError(
                f"The q_z argument (shape: {q_z.shape}) is not a row array "
                f"matching stem properties (shape: {stem_shape})"
            )
        elif (z.ndim == 2) and (q_z.shape != (z.size, n_stems)):
            raise ValueError(
                f"The q_z argument (shape: {q_z.shape}) is not a 2D array congruent "
                f"with the broadcasted shape of the z argument (shape: {z.shape}) "
                f"and stem property arguments (shape: {stem_shape})"
            )

    return


def _validate_demography_array_arguments(
    stem_args: dict[str, NDArray],
    size_args: dict[str, NDArray],
) -> None:
    """Shared validation for T model function inputs.

    Args:
        stem_args: A dictionary of row arrays representing trait values, keyed by the
            trait names.
        size_args: A list of arrays representing points in the stem size and growth
            allometries at which to evaluate functions.
    """

    # Check PFT inputs are all equal sized 1D row arrays.
    try:
        pft_args_shape = check_input_shapes(*stem_args.values())
    except ValueError:
        raise ValueError(
            f"Stem arguments are not of equal length: {','.join(stem_args.keys())}"
        )

    if len(pft_args_shape) > 1:
        raise ValueError(
            f"Stem arguments are not 1D arrays of trait "
            f"values {','.join(stem_args.keys())}"
        )

    # Check size inputs
    try:
        size_args_shape = check_input_shapes(*size_args.values())
    except ValueError:
        raise ValueError(
            f"Arrays of size data are not congruent: {','.join(size_args.keys())}"
        )

    # Explicitly check to see if the size arrays are row arrays and - if so - enforce
    # that they are the same length.
    if size_args:
        if len(size_args_shape) == 1 and not pft_args_shape == size_args_shape:
            raise ValueError("Stem and size inputs are row arrays of unequal length.")

        # Otherwise use np.broadcast_shapes to catch issues
        try:
            _ = np.broadcast_shapes(pft_args_shape, size_args_shape)
        except ValueError:
            raise ValueError("Stem and size inputs are not congruent.")
