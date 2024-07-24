"""Utility classes for deserialising raw JSON data.

Utility classes for deserialising PFT and community data from files containing
JSON arrays.
"""

import json
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from pyrealm.canopy_model.model.cohort import Cohort
from pyrealm.canopy_model.model.community import Community
from pyrealm.canopy_model.model.flora import Flora, PlantFunctionalType


@dataclass_json
@dataclass
class ImportedCohort:
    """Contains raw cohort data which can be loaded from JSON."""

    pft_name: str
    dbh: float
    number_of_members: int


@dataclass_json
@dataclass
class ImportedCommunity:
    """Contains raw community data which can be loaded from JSON."""

    cell_id: int
    cohorts: list[ImportedCohort]


class PlantFunctionalTypeDeserialiser:
    """Utility class for deserialising json file to a list of plant functional types."""

    @classmethod
    def load_plant_functional_types(cls, path: str) -> list[PlantFunctionalType]:
        """Loads a list of plant functional type objects from a json file.

        Uses the built-in json_dataclass functionality to deserialise a json array of
        plant functional type objects to a python list of plant functional types.
        Automatically validates type of the input using a marshmallow type schema.

        :param path: The path for a file containing a json array of Plant Functional
        Type objects.
        :return: A list of deserialised PlantFunctionalType objects.
        """
        with open(path) as file:
            pfts_json = json.load(file)
        pfts = PlantFunctionalType.schema().load(pfts_json, many=True)  # type: ignore[attr-defined]
        return pfts


class CommunityDeserialiser:
    """Provides utility for deserialising json file to list of Community objects."""

    def __init__(
        self, plant_functional_types: list[PlantFunctionalType], cell_area: int
    ):
        self.plant_functional_types: list[PlantFunctionalType] = plant_functional_types
        self.cell_area: float = cell_area

    def load_communities(self, path: str) -> list[Community]:
        """Loads a list of community objects from a json file.

        Uses the built-in json_dataclass functionality to deserialise a json array of
        community objects to a python list of community objects. Automatically
        validates type of the input using a marshmallow type schema.
        :param path: The path for a file containing a json array of Community objects.
        :return: A deserialised list of community objects.
        """

        with open(path) as file:
            communities_json = json.load(file)

        imported_communities = ImportedCommunity.schema().load(
            communities_json, many=True
        )  # type: ignore[attr-defined] # noqa: E501

        flora = Flora(self.plant_functional_types)
        communities = list(
            map(
                lambda imported_community: self.convert_imported_community_to_community(
                    imported_community, flora
                ),
                imported_communities,
            )
        )

        return communities

    def convert_imported_community_to_community(
        self, imported_community: ImportedCommunity, flora: Flora
    ) -> Community:
        """Convert ImportedCommunity to Community object.

        Converts a raw ImportedCommunity object into a Community object with extra
        fields initialised.
        :param imported_community: community imported from raw JSON data.
        :param flora: a flora object containing a dictionary of plant functional types,
        keyed by name.
        :return: a Community object.
        """
        cohorts = self.convert_imported_cohorts_to_cohorts(
            imported_community.cohorts, flora
        )
        return Community(imported_community.cell_id, self.cell_area, cohorts)

    @classmethod
    def convert_imported_cohorts_to_cohorts(
        cls, imported_cohorts: list[ImportedCohort], flora: Flora
    ) -> list[Cohort]:
        """Convert ImportedCohort objects to Cohort objects.

        Converts a list of raw ImportedCohort objects into a list of Cohort objects,
        with extra fields initialised.
        :param imported_cohorts: a list of cohorts imported from raw json data.
        :param flora: a flora object containing a dictionary of plant functional types,
        keyed by name.
        :return: a list of Cohort objects.
        """
        return list(
            map(
                lambda imported_cohort: Cohort(
                    imported_cohort.dbh,
                    imported_cohort.number_of_members,
                    imported_cohort.pft_name,
                    flora,
                ),
                imported_cohorts,
            )
        )
