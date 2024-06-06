from __future__ import annotations

from pathlib import Path

from numpy.typing import NDArray
import numpy as np

from pyrealm.canopy_model.model.flora import Flora
from pyrealm.canopy_model.model.canopy import Canopy


class Community:
    def __init__(
        self,
        cell_id: int,
        cell_area: float,
        cohort_dbh: NDArray[np.float],
        cohort_individuals: NDArray[np.int],
        cohort_pfts: NDArray[np.str],
        flora: Flora,
    ):
        self.cell_id = cell_id
        self.cell_area = cell_area
        self.cohort_dbh = cohort_dbh
        self.cohort_individuals = cohort_individuals
        self.cohort_pfts = cohort_pfts
        self.flora = flora

        # Arrays of required PFT traits for cohorts
        self.traits = CommunityTraits(self.cohort_pfts, self.flora)

        # Calculate geometries from initial dbh
        self.tmodel_geometries = TModelGeometries(self.cohort_dbh, self.traits)

    @classmethod
    def from_json(
        cls, path: Path, cell_id: int, cell_area: float, flora: Flora
    ) -> Community:
        """Factory method creating a Community from a JSON file."""

        cohort_pfts, cohort_individuals, cohort_dbh = do_a_thing_with(path)

        return Community(
            cell_area=cell_area,
            cell_id=cell_id,
            cohort_dbh=cohort_dbh,
            cohort_individuals=cohort_individuals,
            cohort_pfts=cohort_pfts,
            flora=flora,
        )

    def set_canopy(self) -> None:
        """Or this could return a Canopy instance."""

        self.canopy = Canopy(...)


class CommunityTraits:
    def __init__(self, cohort_pfts: NDArray[np.str], flora: Flora):
        self.qm = self.get_trait_array("qm", cohort_pfts, flora)
        ...

    def get_trait_array(
        self, trait_name: str, cohort_pfts: NDArray[np.str], flora: Flora
    ) -> NDArray:
        """Extract array of values across cohorts for a given trait.

        Args:
            trait_name: name
            ...
        """

        return np.array(
            [getattr(flora[this_pft], trait_name) for this_pft in cohort_pfts]
        )


class TModelGeometries:
    """Calculate T Model Geometries.

    Calculate the geometry and mass of cohorts with a given diameter at breast height
    and traits.
    """

    def __init__(self, dbh: NDArray[np.float], traits: CommunityTraits) -> None:
        self.height = traits.h_max * (1 - np.exp(-traits.a_hd * dbh / traits.h_max))

        # Crown area of tree, Equation (8) of Li ea.
        self.crown_area = (
            (np.pi * traits.ca_ratio / (4 * traits.a_hd)) * dbh * self.height
        )

        # Crown fraction, Equation (11) of Li ea.
        self.crown_fraction = self.height / (traits.a_hd * dbh)

        # Masses
        self.mass_stem = (np.pi / 8) * (dbh**2) * self.height * traits.rho_s
        self.mass_fol = self.crown_area * traits.lai * (1 / traits.sla)
        self.mass_swd = (
            self.crown_area
            * traits.rho_s
            * self.height
            * (1 - self.crown_fraction / 2)
            / traits.ca_ratio
        )
