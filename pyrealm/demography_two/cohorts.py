"""TODO: Docs.

Document this.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyrealm.demography.core import CohortMethods, PandasExporter
from pyrealm.demography_two.flora import Flora


def cohort_id_generator(
    mode: Literal["uuid"] | Literal["seqint"] | Literal["seqintstr"] = "uuid",
    fmt: str = "C_{id:06}",
) -> Iterator[str | int]:
    """Generator function for unique cohort IDs.

    Args:
        mode: Use UUID4, sequential integer or formatted sequential integer string
            cohort IDs.
        fmt: A format string for sequential integer string IDs.
    """

    id = 0

    while True:
        if mode == "seqint":
            yield id
            id += 1
        if mode == "seqintstr":
            yield fmt.format(id=id)
            id += 1
        else:
            yield str(uuid.uuid4())


class Cohorts(PandasExporter, CohortMethods):
    """A class to hold data for a set of plant cohorts.

    The provided cohort data must use plant functional types (PFTs) from the provided
    flora and the resulting cohort data merges the PFT data onto the cohorts to provide
    a dataframe of cohort data and traits for each cohort.

    Args:
        flora: A Flora instance providing the PFT data for the cohorts.
        pft_name: The name of the plant functional type for the cohort.
        dbh_value: The diameter at breast height of individuals in the cohort.
        n_individuals: The number of individuals in the cohort.
        community_id: An optional field grouping cohorts into communities.
        cid_generator: A generator providing unique cohort ids.
    """

    def __init__(
        self,
        flora: Flora,
        pft_name: NDArray[np.str_],
        dbh_value: NDArray[np.floating],
        n_individuals: NDArray[np.integer],
        community_id: NDArray[np.integer | np.str_] | None = None,
        cid_generator: Iterator = cohort_id_generator(),
    ) -> None:
        """Init method for Cohorts."""
        # Define attributes
        self.flora: pd.DataFrame = flora.to_dataframe()
        """The flora used with the Cohorts instance, as a pandas dataframe."""
        self._cid_generator = cid_generator
        """A cohort ID generator instance."""
        self.cohorts: pd.DataFrame
        """A pandas dataframe containing the cohort data."""
        self.n_cohorts: int
        """Number of cohorts in the instance."""

        # Validate the inputs - originally did this with pydantic, but support for numpy
        # and bypassing validation when using pydantic to load just ended up tying the
        # code in knots.
        required = [pft_name, dbh_value, n_individuals]
        if community_id is not None:
            required.append(community_id)

        # Do not use check_input_shapes here - we do not want to allow scalar arrays to
        # mix with longer arrays
        shapes = {arr.shape for arr in required}
        if len(shapes) > 1:
            raise ValueError("All arrays must be of the same size")

        shape = next(iter(shapes))
        if len(shape) > 1:
            raise ValueError("Inputs must be 1 dimensional arrays")

        unknown_pfts = set(pft_name).difference(self.flora["name"])
        if unknown_pfts:
            raise ValueError(
                f"PFTs in cohort data not present in flora: {','.join(unknown_pfts)}"
            )

        if np.any(dbh_value <= 0):
            raise ValueError("DBH values must be strictly positive")

        if (not np.issubdtype(n_individuals.dtype, np.integer)) or np.any(
            n_individuals <= 0
        ):
            raise ValueError("The number of individuals must be positive integers")

        columns = {
            "pft_name": pft_name,
            "dbh_value": dbh_value,
            "n_individuals": n_individuals,
        }
        if community_id is not None:
            columns["community_id"] = community_id

        # convert to pandas
        cohorts_df = pd.DataFrame(columns)
        self.n_cohorts = cohorts_df.shape[0]

        cohorts = cohorts_df.merge(self.flora, left_on="pft_name", right_on="name")
        cohorts["cohort_id"] = [
            next(self._cid_generator) for idx in range(cohorts.shape[0])
        ]
        self.cohorts = cohorts

    @classmethod
    def from_csv(
        cls, path: Path, flora: Flora, cid_generator: Iterator = cohort_id_generator()
    ) -> Cohorts:
        """Generate a Cohort instance from a CSV file.

        The cohort data provided is validated before being used to generate the Cohorts
        instance.

        Args:
            path: Path to a CSV file of cohort data.
            flora: A Flora instance providing the PFT data for the cohorts.
            cid_generator: A generator providing unique cohort ids.
        """

        try:
            data = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.ParserError) as excep:
            raise excep

        required_fields = {"pft_name", "dbh_value", "n_individuals"}
        missing_fields = required_fields.difference(data.columns)
        if missing_fields:
            raise ValueError(f"Missing required fields: {','.join(missing_fields)}")

        if "community_id" in data.columns:
            required_fields.add("community_id")

        kwargs = {var: data[var].to_numpy() for var in required_fields}

        return cls(flora=flora, **kwargs, cid_generator=cid_generator)

    def __repr__(self) -> str:
        return f"Cohorts: Data for {self.n_cohorts} cohorts"
