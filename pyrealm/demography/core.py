"""This module provides shared functionality for the :mod:`~pyrealm.demography` module.

* :class:`~pyrealm.demography.core.ToDataFrameMixin` is a class mixin that provides the
  provides the utility :meth:`~pyrealm.demography.core.ToDataFrameMixin.to_dataframe`
  method for generating a dataframe of attributes from demography classes for plotting
  and exploring data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ToDataFrameMixin:
    """Mixin providing a to_dataframe method.

    Classes using this mixin must:

    1. Define a class attribute ``_array_attrs`` that identifies a set of class
       attributes that are all numpy arrays of equal shape.
    2. Have an ``_ndims`` attribute giving the dimensionality of the array attributes.

    The mixin provides the `to_dataframe` method that exports the array attributes as a
    dataframe.

    TODO::

        Vague plan here that this could also support polars or simply swap to polars for
        increased performance without an API change.
    """

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the instance array attributes into a data frame."""
        try:
            array_attrs = getattr(self, "_array_attrs")
            ndims = getattr(self, "_ndims")
        except KeyError:
            raise RuntimeError("ToDataFrameMixin used with incompatible class.")

        # Extract the attributes into a dictionary
        data = {k: getattr(self, k) for k in array_attrs}

        # Simple 1D case
        if ndims == 1:
            return pd.DataFrame(data)

        # Otherwise ravel the attribute data to give 1D values
        return pd.DataFrame({ky: np.ravel(vl) for ky, vl in data.items()})
