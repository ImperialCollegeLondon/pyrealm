"""TODO: Docs.

Document this.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyrealm.demography_two.flora import Flora


def cohort_id_generator(
    mode: Literal["uuid"] | Literal["int"] | Literal["str"] = "uuid",
    str_fmt: str = "C_{id:06}",
) -> Iterator[str | int]:
    """Generator function for unique cohort IDs.

    Args:
        mode: Use UUID4, sequential integer or formatted sequential integer string
            cohort IDs.
        str_fmt: A format string for sequential integer string IDs.
    """

    id = 0

    while True:
        if mode == "int":
            yield id
            id += 1
        if mode == "str":
            yield str_fmt.format(id=id)
            id += 1
        else:
            yield str(uuid.uuid4())


class Cohorts(pd.DataFrame):
    """The Cohorts class.

    This is simply a data frame.
    """

    pass


def create_cohorts(
    flora: Flora,
    cid_generator: Iterator,
    pft_name: NDArray[np.str_],
    dbh_value: NDArray[np.floating],
    n_individuals: NDArray[np.integer],
    community_id: NDArray[np.integer | np.str_] | None = None,
) -> Cohorts:
    """Init method for Cohorts."""

    # Define attributes
    flora_df = flora.to_dataframe()

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

    unknown_pfts = set(pft_name).difference(flora.name)
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
    cohorts_df = Cohorts(columns)

    cohorts = cohorts_df.merge(flora_df, left_on="pft_name", right_on="name")
    cohorts.insert(0, "cohort_id", [next(cid_generator) for idx in range(shape[0])])

    return cohorts


def create_cohorts_from_csv(
    path: Path, flora: Flora, cid_generator: Iterator
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

    return create_cohorts(flora=flora, **kwargs, cid_generator=cid_generator)
