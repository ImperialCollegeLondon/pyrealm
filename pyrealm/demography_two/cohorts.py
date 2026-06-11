"""TODO: Docs.

Document this.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, model_validator

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


class CohortData(BaseModel):
    """A pydantic validation model for CohortData.

    The model enforces a set of required fields (pft_name, dbh_value and n_individuals)
    and applies some validation (lists of value of equal length, numeric values greater
    than zero). The optional `community_id` field
    """

    pft_name: list[str]
    r"""The name of the plant functional type for the cohort."""
    dbh_value: list[Annotated[float, Field(gt=0)]]
    r"""The diameter at breast height of individuals in the cohort."""
    n_individuals: list[Annotated[int, Field(gt=0)]]
    r"""The number of individuals in the cohort."""
    community_id: list[Any] | None = None
    r"""An optional field grouping cohorts into communities."""

    # TODO think about cell_id alias?
    # = Field(validation_alias=AliasChoices('community_id', 'cell_id'))

    @model_validator(mode="after")
    def model_validation(self, info: ValidationInfo) -> Self:
        """Checks all fields are of equal length."""

        # Check field lengths of provided data
        field_lengths = set([len(getattr(self, nm)) for nm in self.model_fields_set])
        if len(field_lengths) > 1:
            raise ValueError(
                f"Unequal field lengths: {', '.join([str(it) for it in field_lengths])}"
            )

        return self

    @classmethod
    def _from_file_data(cls, file_data: dict, strict: bool = False) -> CohortData:
        """Create a CohortData object from a dictionary of data.

        Args:
            file_data: The payload from a data file defining plant functional types.
            strict: Require that all traits are specified in the input data.
        """
        try:
            cohort = cls.model_validate(file_data)
        except ValidationError as excep:
            raise excep

        return cohort

    @classmethod
    def from_csv(cls, path: Path) -> CohortData:
        """Create a CohortData object from a CSV file.

        Args:
            path: A path to a CSV file of cohort data.
        """

        try:
            data = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.ParserError) as excep:
            raise excep

        return cls._from_file_data(data.to_dict(orient="list"))

    def to_dataframe(self) -> pd.DataFrame:
        """Return a CohortData instance as a pandas DataFrame."""

        data = self.model_dump()

        # Do not broadcast community id = None to a series. If no community_id data was
        # provided when the instance was created, then drop it.
        if data["community_id"] is None:
            del data["community_id"]

        return pd.DataFrame(data)


class Cohorts(PandasExporter, CohortMethods):
    """A class to hold data for a set of plant cohorts.

    The provided cohort data must use plant functional types (PFTs) from the provided
    flora and the resulting cohort data merges the PFT data onto the cohorts to provide
    a dataframe of cohort data and traits for each cohort.

    Args:
        cohort_data: An instance of CohortData providing validated cohort data.
        flora: A Flora instance providing the PFT data for the cohorts.
    """

    def __init__(
        self,
        cohort_data: CohortData,
        flora: Flora,
        cid_generator: Iterator = cohort_id_generator(),
    ) -> None:
        self.flora: pd.DataFrame = flora.to_dataframe()
        """The flora used with the Cohorts instance, as a pandas dataframe."""
        self.cohorts: pd.DataFrame
        """A pandas dataframe containing the cohort data."""
        self._cid_generator = cid_generator
        """A cohort ID generator instance."""

        cohorts_df = cohort_data.to_dataframe()

        unknown_pfts = set(cohorts_df["pft_name"]).difference(self.flora["name"])
        if unknown_pfts:
            raise ValueError(
                f"PFTs in cohort data not present in flora: {','.join(unknown_pfts)}"
            )

        cohorts = cohorts_df.merge(self.flora, left_on="pft_name", right_on="name")
        cohorts["cohort_id"] = [
            next(self._cid_generator) for idx in range(cohorts.shape[0])
        ]
        self.cohorts = cohorts

    @classmethod
    def from_csv(cls, path: Path, flora: Flora) -> Cohorts:
        """Generate a Cohort instance from a CSV file.

        The cohort data provided is validated before being used to generate the Cohorts
        instance.

        Args:
            path: Path to a CSV file of cohort data.
            flora: A Flora instance providing the PFT data for the cohorts.
        """

        cohort_data = CohortData.from_csv(path)

        return cls(cohort_data=cohort_data, flora=flora)
