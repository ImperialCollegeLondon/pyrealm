"""Contains classes relating to modelling a cohort."""

from pyrealm.canopy_model.model.flora import Flora, PlantFunctionalType
from pyrealm.canopy_model.model.jaideep_t_model_extension import (
    CanopyFactors,
    calculate_stem_canopy_factors,
)
from pyrealm.canopy_model.model.t_model import (
    TModelGeometry,
    calculate_t_model_geometry,
)


class Cohort:
    """Contains cohort data.

    Stores cohort data, initialises the detailed plant functional data and calculates
    the T Model Geometry of a given cohort.
    """

    def __init__(self, dbh: float, number_of_members: int, pft_name: str, flora: Flora):
        self.dbh: float = dbh
        self.number_of_members: int = number_of_members
        self.pft: PlantFunctionalType = self.__look_up_plant_functional_type(
            pft_name, flora
        )
        self.t_model_geometry: TModelGeometry = calculate_t_model_geometry(
            self.dbh, self.pft
        )
        self.canopy_factors: CanopyFactors = calculate_stem_canopy_factors(
            self.pft, self.t_model_geometry
        )

    @classmethod
    def __look_up_plant_functional_type(
        cls, pft_name: str, flora: Flora
    ) -> PlantFunctionalType:
        """Retrieve plant functional type for a cohort from the flora dictionary."""
        pft = flora.get(pft_name)

        if pft is None:
            raise Exception("Cohort data supplied with in an invalid PFT name.")
        return pft
