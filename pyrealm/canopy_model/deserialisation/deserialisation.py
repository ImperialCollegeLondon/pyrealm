"""Utility classes for deserialising raw JSON data.

Utility classes for deserialising PFT and community data from files containing
JSON arrays.
"""

import json

from pyrealm.canopy_model.model.flora import PlantFunctionalType


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
