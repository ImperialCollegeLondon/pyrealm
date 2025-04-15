"""This module provides shared functionality for the :mod:`~pyrealm.demography` module,
implementing two abstract base classes that are used to share core methods across
demography classes:

* :class:`~pyrealm.demography.core.PandasExporter` provides the utility
  :meth:`~pyrealm.demography.core.PandasExporter.to_pandas` method for extracting data
  from demography classes for plotting and exploring data.
* :class:`~pyrealm.demography.core.CohortMethods` provides the utility
  :meth:`~pyrealm.demography.core.CohortMethods.add_cohort_data` and
  :meth:`~pyrealm.demography.core.CohortMethods.drop_cohort_data` methods that are used
  to append new cohort data across some demography dataclasses.
"""  # noqa: D205, D415

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
    names a set of instance attributes that are all numpy arrays of equal length. They
    must also define the class attribute ``count_attr`` that identifies an instance
    attribute used to records the number of cohorts or stems in the class.

    The class then inherit:

    * The `add_cohorts` method, which allows a second instance of the same class to be
      joined to the calling instance, concatenting each of the array attributes from
      the second instance onto the calling instance and updating ``n_cohorts``.
    * The `drop_cohorts` method, which takes a set of indices onto the array attributes
      and drops the values from those indices for each array attribute and updating
      ``n_cohorts``.
    """

    array_attrs: ClassVar[tuple[str, ...]]
    count_attr: ClassVar[str]

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
        # instance. This need to respect the attribute array dimensions. If the
        # attribute is one dimensional (e.g. traits), then concatenate on axis=0, but
        # for 2 traits, need to concatenate on axis 1 to extend the trait axis.
        for trait in self.array_attrs:
            current = getattr(self, trait)
            setattr(
                self,
                trait,
                np.concatenate(
                    [current, getattr(new_data, trait)], axis=(current.ndim - 1)
                ),
            )

        # Update the count attribute
        setattr(
            self,
            self.count_attr,
            getattr(self, self.count_attr) + getattr(new_data, self.count_attr),
        )

    def drop_cohort_data(self, drop_indices: NDArray[np.int_]) -> None:
        """Drop array attribute values from an instance.

        Args:
            drop_indices: An array of integer indices to drop from each array attribute.
        """

        # Drop from each trait along the last dimension - handles 2D height x stem and
        # 1D stem traits.

        for trait in self.array_attrs:
            current = getattr(self, trait)
            setattr(
                self, trait, np.delete(current, drop_indices, axis=(current.ndim - 1))
            )

        # Update the count attribute
        setattr(
            self,
            self.count_attr,
            getattr(self, self.count_attr) - len(drop_indices),
        )


def _validate_demography_array_arguments(
    trait_args: dict[str, NDArray],
    size_args: dict[str, NDArray] = {},
    at_size_args: dict[str, NDArray] = {},
) -> None:
    """Shared validation for demography inputs.

    Trait arguments of functions should always be 1 dimensional arrays (row arrays) or
    arrays of size 1 ('scalar'), representing trait values that are constant within a
    cohort or stem. If multiple traits are being validated, then they should all have
    have the same shape or be scalar.

    In addition to simple validation of trait inputs, many demographic functions make
    predictions of stem allometries, allocation and crown profile at a range of sizes.
    These size arguments provide values at which to estimate predictions across stems
    with different traits and  could represent different stem sizes (e.g. dbh) or
    different stem heights at which to evaluate crown profiles. If size arguments are
    provided, then they are also validated. All size arguments must have the same shape
    or be scalar. Given that the stem traits provide N values, the shape of the size
    arguments can then be:

    * A scalar array (i.e. with size 1, such as np.array(1) or np.array([1])), that
      provides a single size at which to calculate predictions for all N stems.
    * A one dimensional array with identical shape to the stem properties (``(N,)``)
      that provides individual single values for each stem.
    * A two dimensional column vector (i.e. with shape ``(M, 1)``) that provides a set
      of ``M`` values at which to calculate multiple predictions for all stem traits.
    * A two-dimensional array ``(M, N)`` that provides individual predictions for each
      stem.

    Lastly, some functions generate predictions for stems at a particular size, given a
    third input. For example, the stem allocation process evaluates how potential GPP is
    allocated for a stem of a given size. (TBD - add crown example). If provided, these
    ``at_size_args`` values are validated as follows:

    * if ``z`` is a row array, ``q_z`` must then have identical shape, or
    * if ``z`` is a column array ``(N, 1)``, ``q_z`` must then have shape ``(N,
        n_stem_properties``).

    Args:
        trait_args: A dictionary of row arrays representing trait values, keyed by the
            trait names.
        size_args: A dictionary of arrays representing size values for stem allometry or
            canopy height at which to evaluate functions, keyed by the value name.
        at_size_args: A dictionary of arrays providing values that are to be evaluated
            given predictions for a set of traits at a given size and which must be
            congruent with both trait and size arguments. The dictionary should be keyed
            with the value name.
    """

    # NOTE - this validation deliberately does not use check_input_shapes because it is
    # insisting on _identical_ shapes or scalar arrays is too restrictive. See
    # discussion in https://github.com/ImperialCollegeLondon/pyrealm/pull/342

    # Check PFT inputs are all equal sized 1D row arrays or a mix of 1D rows and scalar
    # values
    try:
        trait_args_shape = np.broadcast_shapes(
            *[arr.shape for arr in trait_args.values()]
        )
    except ValueError:
        raise ValueError(
            f"Trait arguments are not equal shaped or "
            f"scalar: {','.join(trait_args.keys())}"
        )

    if len(trait_args_shape) > 1:
        raise ValueError(
            f"Trait arguments are not 1D arrays: {','.join(trait_args.keys())}"
        )

    # Check the size inputs are broadcastable if they are provided.
    if size_args:
        try:
            size_args_shape = np.broadcast_shapes(
                *[arr.shape for arr in size_args.values()]
            )
        except ValueError:
            raise ValueError(
                f"Size arguments are not equal shaped or "
                f"scalar: {','.join(size_args.keys())}"
            )

        # Test whether the shape of the size args are compatible with the traits args
        try:
            trait_size_shape = np.broadcast_shapes(trait_args_shape, size_args_shape)
        except ValueError:
            raise ValueError(
                f"The array shapes of the trait {trait_args_shape} and "
                f"size {size_args_shape} arguments are not congruent."
            )

    # Now check at size args if provided
    if at_size_args and not size_args:
        raise ValueError("Only provide `at_size_args` when `size_args` also provided.")

    if at_size_args:
        # Are the at_size values broadcastable?
        try:
            at_size_args_shape = np.broadcast_shapes(
                *[arr.shape for arr in at_size_args.values()]
            )
        except ValueError:
            raise ValueError(
                f"At size arguments are not equal shaped or "
                f"scalar: {','.join(at_size_args.keys())}"
            )

        # Are they congruent with the trait and size values.
        try:
            _ = np.broadcast_shapes(trait_size_shape, at_size_args_shape)
        except ValueError:
            raise ValueError(
                f"The broadcast shapes of the trait and size arguments "
                f"{trait_size_shape} are not congruent with the shape of the at_size "
                f"arguments {at_size_args_shape}."
            )


def _enforce_2D(array: NDArray) -> NDArray:
    """Utility conversion to force two dimensional outputs.

    Depending on the input dimensions, the calculations in the T Model and other
    demography models can return scalar, one dimensional or two dimensional arrays. This
    utility function is used to give a consistent output array dimensionality.

    Args:
        array: A numpy array
    """

    match array.ndim:
        case 0:
            return array[None, None]
        case 1:
            return array[None, :]
        case 2:
            return array
        case _:
            raise ValueError("Demography array of more than 2 dimensions.")
