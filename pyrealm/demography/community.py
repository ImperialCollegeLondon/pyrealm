"""Contains a class representing properties of a community."""

from __future__ import annotations

# import json
# import sys
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from marshmallow import Schema, fields, validates_schema
from marshmallow.exceptions import ValidationError
from numpy.typing import NDArray

from pyrealm.demography import t_model_functions as t_model
from pyrealm.demography.flora import Flora

# if sys.version_info[:2] >= (3, 11):
#     import tomllib
#     from tomllib import TOMLDecodeError
# else:
#     import tomli as tomllib
#     from tomli import TOMLDecodeError


class CommunitySchema(Schema):
    """A validation schema for community initialisation data.

    This schema can be used to validate the data being used to create a Community
    instance via one of the factory methods. It does not validate the Flora argument,
    which is loaded and validated separately.
    """

    cell_id = fields.Integer(required=True)
    cell_area = fields.Float(required=True)
    cohort_dbh_values = fields.List(fields.Float(), required=True)
    cohort_n_individuals = fields.List(fields.Integer(), required=True)
    cohort_pft_names = fields.List(fields.Str(), required=True)

    @validates_schema
    def validate_array_lengths(self, data: dict, **kwargs: Any) -> None:
        """Schema wide validation.

        This checks that the cohort data arrays are of equal length.

        Args:
            data: Data passed to the validator
            kwargs: Additional keyword arguments passed by marshmallow
        """

        len_dbh = len(data["cohort_dbh_values"])
        len_n = len(data["cohort_n_individuals"])
        len_pft = len(data["cohort_pft_names"])

        if not ((len_dbh == len_n) and (len_dbh == len_pft)):
            raise ValidationError("Cohort arrays of unequal length.")


@dataclass
class Community:
    """Class containing properties of a community.

    A community is a group of plants in a given location. A location consists of a cell
    with a specified area and ID.

    A community is broken down into cohorts, ie a collection of plants with the same
    diameter at breast height (DBH) and plant functional type (PFT). This is
    represented inside the class as a struct of arrays, with each element in a given
    array representing a property of a cohort. The properties of a given cohort are
    spread across the arrays and associated positionally, e.g. a cohort that has its
    pft_h_max in the third element of the pft_h_max_values array will have the number
    of individuals in the cohort in the third element of the
    cohort_number_of_individuals array. Care must therefore be taken to
    modify all the arrays when adding and removing cohorts.

    In addition to the properties the class is initialised with, the following
    properties are calculated during initialisation and exposed publicly:
    geometry of cohorts calculated using the t model, and canopy factors from
    Jaideep's extension to the t model.

    A method is also provided to load community data from a csv file.
    """

    # Dataclass attributes for initialisation
    # - community wide properties
    cell_id: int
    cell_area: float
    flora: Flora

    # - arrays representing properties of cohorts
    cohort_dbh_values: InitVar[NDArray[np.float32]]
    cohort_n_individuals: InitVar[NDArray[np.int_]]
    cohort_pft_names: InitVar[NDArray[np.str_]]

    # Post init properties
    number_of_cohorts: int = field(init=False)

    # Dataframe of cohort
    cohort_data: pd.DataFrame = field(init=False)

    def __post_init__(
        self,
        cohort_dbh_values: NDArray[np.float32],
        cohort_n_individuals: NDArray[np.int_],
        cohort_pft_names: NDArray[np.str_],
    ) -> None:
        """Populate derived community attributes.

        The ``__post_init__`` builds a pandas dataframe of PFT values and T model
        predictions across the initial cohort data.
        """

        # Check the initial PFT values are known
        unknown_pfts = set(cohort_pft_names).difference(self.flora.keys())

        if unknown_pfts:
            raise ValueError(
                f"Plant functional types unknown in flora: {','.join(unknown_pfts)}"
            )

        # Check the cohort inputs are of equal length
        if not (
            (cohort_dbh_values.shape == cohort_n_individuals.shape)
            and (cohort_dbh_values.shape == cohort_pft_names.shape)
        ):
            raise ValueError("Cohort data are not equally sized")

        # Convert to a dataframe
        cohort_data = pd.DataFrame(
            {
                "name": cohort_pft_names,
                "dbh": cohort_dbh_values,
                "n_individuals": cohort_n_individuals,
            }
        )
        # Broadcast the pft trait data to the cohort data by merging with the flora data
        # and then store as the cohort data attribute
        self.cohort_data = pd.merge(cohort_data, self.flora.data)
        self.number_of_cohorts = self.cohort_data.shape[0]

        # Populate the T model fields
        self._calculate_t_model()

    def _calculate_t_model(self) -> None:
        """Calculate T Model predictions across cohort data.

        This method populates or updates the community attributes predicted by the T
        Model :cite:`Li:2014bc` and by the canopy shape extensions to the T Model
        implemented in PlantFate :cite:`joshi:2022a`.
        """

        # Add data to cohort dataframes capturing the T Model geometry
        # - Classic T Model scaling
        self.cohort_data["height"] = t_model.calculate_heights(
            h_max=self.cohort_data["h_max"],
            a_hd=self.cohort_data["a_hd"],
            dbh=self.cohort_data["dbh"],
        )

        self.cohort_data["crown_area"] = t_model.calculate_crown_areas(
            ca_ratio=self.cohort_data["ca_ratio"],
            a_hd=self.cohort_data["a_hd"],
            dbh=self.cohort_data["dbh"],
            height=self.cohort_data["height"],
        )

        self.cohort_data["crown_fraction"] = t_model.calculate_crown_fractions(
            a_hd=self.cohort_data["a_hd"],
            dbh=self.cohort_data["dbh"],
            height=self.cohort_data["height"],
        )

        self.cohort_data["stem_mass"] = t_model.calculate_stem_masses(
            rho_s=self.cohort_data["rho_s"],
            dbh=self.cohort_data["dbh"],
            height=self.cohort_data["height"],
        )

        self.cohort_data["foliage_mass"] = t_model.calculate_foliage_masses(
            sla=self.cohort_data["sla"],
            lai=self.cohort_data["lai"],
            crown_area=self.cohort_data["crown_area"],
        )

        self.cohort_data["sapwood_mass"] = t_model.calculate_sapwood_masses(
            rho_s=self.cohort_data["rho_s"],
            ca_ratio=self.cohort_data["ca_ratio"],
            height=self.cohort_data["height"],
            crown_area=self.cohort_data["crown_area"],
            crown_fraction=self.cohort_data["crown_fraction"],
        )

        # Canopy shape extension to T Model from PlantFATE
        self.cohort_data["canopy_z_max"] = t_model.calculate_canopy_z_max(
            z_max_prop=self.cohort_data["z_max_prop"],
            height=self.cohort_data["height"],
        )
        self.cohort_data["canopy_r0"] = t_model.calculate_canopy_r0(
            q_m=self.cohort_data["q_m"],
            crown_area=self.cohort_data["crown_area"],
        )

    @classmethod
    def _from_file_data(cls, flora: Flora, file_data: dict) -> Community:
        """Create a Flora object from a JSON string.

        Args:
            flora: The Flora instance to be used with the community
            file_data: The payload from a data file defining plant functional types.
        """

        # Validate the input data against the schema.
        try:
            community_data = CommunitySchema().load(data=file_data)  # type: ignore[attr-defined]
        except ValidationError as excep:
            raise excep

        # Pass validated data into class instance
        return cls(
            cell_id=community_data["cell_id"],
            cell_area=community_data["cell_area"],
            cohort_dbh_values=np.array(community_data["cohort_dbh_values"]),
            cohort_n_individuals=np.array(community_data["cohort_n_individuals"]),
            cohort_pft_names=np.array(community_data["cohort_pft_names"]),
            flora=flora,
        )

    # @classmethod
    # def from_json(cls, path: Path) -> Flora:
    #     """Create a Flora object from a JSON file.

    #     Args:
    #         path: A path to a JSON file of plant functional type definitions.
    #     """

    #     try:
    #         file_data = json.load(open(path))
    #     except (FileNotFoundError, json.JSONDecodeError) as excep:
    #         raise excep

    #     return cls._from_file_data(file_data=file_data)

    # @classmethod
    # def from_toml(cls, path: Path) -> Flora:
    #     """Create a Flora object from a TOML file.

    #     Args:
    #         path: A path to a TOML file of plant functional type definitions.
    #     """

    #     try:
    #         file_data = tomllib.load(open(path, "rb"))
    #     except (FileNotFoundError, TOMLDecodeError) as excep:
    #         raise excep

    #     return cls._from_file_data(file_data)

    # @classmethod
    # def from_csv(cls, path: Path) -> Flora:
    #     """Create a Flora object from a CSV file.

    #     Args:
    #         path: A path to a CSV file of plant functional type definitions.
    #     """

    #     try:
    #         data = pd.read_csv(path)
    #     except (FileNotFoundError, pd.errors.ParserError) as excep:
    #         raise excep

    #     return cls._from_file_data({"pft": data.to_dict(orient="records")})

    # @classmethod
    # def load_communities_from_csv(
    #     cls, cell_area: float, csv_path: str, flora: Flora
    # ) -> list[Community]:
    #     """Loads a list of communities from a csv provided in the appropriate format.

    #     The csv should contain the following columns: cell_id,
    #     diameter_at_breast_height, plant_functional_type, number_of_individuals. Each
    #     row in the csv should represent one cohort.

    #     :param cell_area: the area of the cell at each location, this is assumed to be
    #     the same across all the locations in the csv.
    #     :param csv_path: path to the csv containing community data, as detailed above.
    #     :param flora: a flora object, ie a dictionary of plant functional properties,
    #     keyed by pft name.
    #     :return: a list of community objects, loaded from the csv
    #     file.
    #     """
    #     community_data = pd.read_csv(csv_path)

    #     data_grouped_by_community = community_data.groupby(community_data.cell_id)

    #     communities = []

    #     for cell_id in data_grouped_by_community.groups:
    #         community_dataframe = data_grouped_by_community.get_group(cell_id)
    #         dbh_values = community_dataframe["diameter_at_breast_height"].to_numpy(
    #             dtype=np.float32
    #         )
    #         number_of_individuals = community_dataframe[
    #             "number_of_individuals"
    #         ].to_numpy(dtype=np.int_)
    #         pft_names = community_dataframe["plant_functional_type"].to_numpy(
    #               dtype=str
    #         )
    #         community_object = Community(
    #             cell_id,  # type:ignore
    #             cell_area,
    #             dbh_values,
    #             number_of_individuals,
    #             pft_names,
    #             flora,
    #         )
    #         communities.append(community_object)

    #     return communities
