"""Functionality for community modelling."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyrealm.canopy_model.base_objects import (
    Cohort,
    Flora,
    ImportedCommunity,
    PlantFunctionalType,
)


@dataclass
class TModelGeometry:
    """Cohort traits calculated using the T Model.

    Includes the plant functional type of the cohort, and the geometry and mass
    calculated using the T Model. I'm not sure this all belongs in the same class
    this way. Would also like to find a better name for this class.
    """

    height: float
    crown_area: float
    crown_fraction: float
    mass_stem: float
    mass_fol: float
    mass_swd: float


class Community:
    """Beginnings of a community class.

    This could handle multiple locations or a single location.
    """

    def __init__(
        self, flora: Flora, imported_community: ImportedCommunity, cell_area: float
    ) -> None:
        self.flora: Flora = flora
        self.cell_area: float = cell_area
        self.cohorts: list[Cohort] = imported_community.cohorts

        self.pfts: list[PlantFunctionalType] = list(
            map(self.get_plant_functional_type_for_cohort, self.cohorts)
        )

        self.t_model_geometries: list[TModelGeometry] = list(
            map(self.calculate_t_model_geometry, self.cohorts, self.pfts)
        )

        # TODO Note the pfts, cohorts and t_model geometries aren't tied together,
        #  but the order is preserved in the lists, this may become an issue when we
        #  start adding and removing cohorts and may need a rethink.

        # Things to add later

        # pft keyed dictionary of propagule density, size, energy content pft keyed
        # dictionary of propagule density, size, energy content pft keyed dictionary of
        # propagule density, size, energy content
        self.seedbank: object

        # per cohort structure of fruit mass, energy, size?
        self.fruit: object

    def get_plant_functional_type_for_cohort(
        self, cohort: Cohort
    ) -> PlantFunctionalType:
        """Retrieve plant functional type for a cohort from the flora dictionary."""
        pft = self.flora.get(cohort.pft_name)

        if pft is None:
            raise Exception("PFT not provided in list")
        return pft

    @classmethod
    def calculate_t_model_geometry(
        cls, cohort: Cohort, pft: PlantFunctionalType
    ) -> TModelGeometry:
        """placeholder."""

        height = pft.h_max * (1 - np.exp(-pft.a_hd * cohort.dbh / pft.h_max))

        # Crown area of tree, Equation (8) of Li ea.
        crown_area = (np.pi * pft.ca_ratio / (4 * pft.a_hd)) * cohort.dbh * height

        # Crown fraction, Equation (11) of Li ea.
        crown_fraction = height / (pft.a_hd * cohort.dbh)

        # Masses
        mass_stem = (np.pi / 8) * (cohort.dbh**2) * height * pft.rho_s
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

    def recruit(self) -> None:
        """Add new cohorts from the seedbank."""

    pass

    def remove_empty_cohorts(self) -> None:
        """Remove cohort data for empty cohorts."""

        pass

    def calculate_gpp(self, cell_ppfd: NDArray, lue: NDArray) -> None:
        """Estimate the gross primary productivity.

        Not sure where to place this - need an array of LUE that matches to the

        """

        pass
