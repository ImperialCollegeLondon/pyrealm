"""Utility classes for deserialising raw JSON data.

Utility classes for deserialising PFT and community data from files containing
JSON arrays.
"""

import json

from pyrealm.canopy_model.model.community import Community
from pyrealm.canopy_model.model.flora import Flora, PlantFunctionalType


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
        pfts = PlantFunctionalType.schema().load(pfts_json, many=True)  # type: ignore[attr-defined] # noqa: E501
        return pfts


class CommunityDeserialiser:
    """Provides utility for deserialising json file to list of Community objects."""

    def __init__(
        self, plant_functional_types: list[PlantFunctionalType], cell_area: int
    ):
        self.plant_functional_types = plant_functional_types
        self.cell_area = cell_area

    def load_communities(self, path: str) -> list[Community]:
        """Loads a list of community objects from a json file.

        Uses the built-in json_dataclass functionality to deserialise a json array of
        community objects to a python list of community objects. Automatically
        validates type of the input using a marshmallow type schema.
        :param path: The path for a file containing a json array of Community objects.
        :return: A deserialised list of community objects.
        """

        community_schema = Community.schema()  # type: ignore[attr-defined] # noqa: E501

        flora = Flora(self.plant_functional_types)
        community_schema.context = {"flora": flora, "cell_area": self.cell_area}

        with open(path) as file:
            communities_json = json.load(file)

        return community_schema.load(communities_json, many=True)
