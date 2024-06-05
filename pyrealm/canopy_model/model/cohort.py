"""Contains classes relating to modelling a cohort."""

from dataclasses import InitVar, dataclass, field

from dataclasses_json import dataclass_json

from pyrealm.canopy_model.model.flora import Flora, PlantFunctionalType
from pyrealm.canopy_model.model.t_model import (
    TModelGeometry,
    calculate_t_model_geometry,
)


@dataclass_json
@dataclass
class Cohort:
    """Contains cohort data.

    Dataclass for storing Cohort data imported from json, which also initialises the
    detailed plant functional data and calculates the T Model Geometry of a given
    cohort.
    """

    pft_name: InitVar[str]
    flora: InitVar[Flora]  # where to declare this...
    dbh: float
    number_of_members: int
    pft: PlantFunctionalType = field(init=False)
    t_model_geometry: TModelGeometry = field(init=False)

    def __post_init__(self, pft_name: str, flora: Flora) -> None:
        self.pft = self.look_up_plant_functional_type(pft_name, flora)
        self.t_model_geometry = calculate_t_model_geometry(self.dbh, self.pft)

    @classmethod
    def look_up_plant_functional_type(
        cls, pft_name: str, flora: Flora
    ) -> PlantFunctionalType:
        """Retrieve plant functional type for a cohort from the flora dictionary."""
        pft = flora.get(pft_name)

        if pft is None:
            raise Exception("Cohort data supplied with in an invalid PFT name.")
        return pft
