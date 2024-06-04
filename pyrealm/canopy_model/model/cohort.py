"""Contains classes relating to modelling a cohort."""

from dataclasses import InitVar, dataclass, field

import numpy as np
from dataclasses_json import dataclass_json

from pyrealm.canopy_model.model.flora import Flora, PlantFunctionalType


@dataclass
class TModelGeometry:
    """Geometry and mass of a cohort, calculated using the T Model.

    Includes the geometry and mass of a particular cohort with a given
    diameter at breast height, calculated using the T Model.
    """

    height: float
    crown_area: float
    crown_fraction: float
    mass_stem: float
    mass_fol: float
    mass_swd: float


@dataclass_json
@dataclass
class Cohort:
    """Contains cohort data.

    Dataclass for storing Cohort data imported from json, which also initialises the
    detailed plant functional data and calculates the T Model Geometry of a given
    cohort.
    """

    pft_name: InitVar[str]
    flora: InitVar[Flora]
    dbh: float
    number_of_members: int
    pft: PlantFunctionalType = field(init=False)
    t_model_geometry: TModelGeometry = field(init=False)

    def __post_init__(self, pft_name: str, flora: Flora) -> None:
        self.pft = self.look_up_plant_functional_type(pft_name, flora)
        self.t_model_geometry = self.calculate_t_model_geometry(self.dbh, self.pft)

    @classmethod
    def look_up_plant_functional_type(
        cls, pft_name: str, flora: Flora
    ) -> PlantFunctionalType:
        """Retrieve plant functional type for a cohort from the flora dictionary."""
        pft = flora.get(pft_name)

        if pft is None:
            raise Exception("Cohort data supplied with in an invalid PFT name.")
        return pft

    @classmethod
    def calculate_t_model_geometry(
        cls, dbh: float, pft: PlantFunctionalType
    ) -> TModelGeometry:
        """Calculate T Model Geometry.

        Calculate the geometry and mass of a plant or cohort with a given diameter at
        breast height and plant functional type.
        """

        height = pft.h_max * (1 - np.exp(-pft.a_hd * dbh / pft.h_max))

        # Crown area of tree, Equation (8) of Li ea.
        crown_area = (np.pi * pft.ca_ratio / (4 * pft.a_hd)) * dbh * height

        # Crown fraction, Equation (11) of Li ea.
        crown_fraction = height / (pft.a_hd * dbh)

        # Masses
        mass_stem = (np.pi / 8) * (dbh**2) * height * pft.rho_s
        mass_fol = crown_area * pft.lai * (1 / pft.sla)
        mass_swd = (
            crown_area * pft.rho_s * height * (1 - crown_fraction / 2) / pft.ca_ratio
        )
        return TModelGeometry(
            height,
            crown_area,
            crown_fraction,
            mass_stem,
            mass_fol,
            mass_swd,
        )
