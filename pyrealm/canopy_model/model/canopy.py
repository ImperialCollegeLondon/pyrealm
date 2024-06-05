"""Functionality for canopy modelling."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from pyrealm.canopy_model.model.community import Community
from pyrealm.canopy_model.model.jaideep_t_model_extension import (
    calculate_projected_canopy_area_for_individual,
)


class Canopy:
    """placeholder."""

    def __init__(self, community: Community, canopy_gap_fraction: float) -> None:
        """placeholder."""
        self.f_g = canopy_gap_fraction
        self.cohorts = community.cohorts

        self.max_individual_height = max(
            cohort.t_model_geometry.height for cohort in self.cohorts
        )

        self.canopy_layer_heights = self.calculate_canopy_layer_heights(
            community.cell_area, self.f_g
        )

    def calculate_community_projected_area_at_z(self, z: float) -> float:
        """Calculate the total area of community stems."""
        cohort_areas_at_z = map(
            lambda cohort: cohort.number_of_members
            * calculate_projected_canopy_area_for_individual(
                z, cohort.pft, cohort.t_model_geometry, cohort.canopy_factors
            ),
            self.cohorts,
        )

        return sum(cohort_areas_at_z)

    def solve_canopy_closure_height(
        self,
        z: float,
        l: int,
        A: float,
        fG: float,
    ) -> float:
        """Solver function for canopy closure height.

        This function returns the difference between the total community projected area
        at a height $z$ and the total available canopy space for canopy layer $l$, given
        the community gap fraction for a given height. It is used with a root solver to
        find canopy layer closure heights $z^*_l* for a community.
        :param fG: community gap fraction
        :param A: community area
        :param l: layer index
        :param z: height
        """

        community_projected_area_at_z = self.calculate_community_projected_area_at_z(z)

        # Return the difference between the projected area and the available space
        return community_projected_area_at_z - (A * l) * (1 - fG)

    def calculate_canopy_layer_heights(self, A: float, fG: float) -> NDArray:
        """Placeholder."""
        # Calculate the number of layers
        cohort_crown_areas = map(
            lambda cohort: cohort.number_of_members
            * cohort.t_model_geometry.crown_area,
            self.cohorts,
        )
        total_community_crown_area = sum(cohort_crown_areas)
        number_of_layers = int(np.ceil(total_community_crown_area / (A * (1 - fG))))

        # Data store for z*
        z_star = np.zeros(number_of_layers)

        # Loop over the layers TODO - edge case of completely filled final layer
        for n in np.arange(number_of_layers - 1):
            z_star[n] = root_scalar(
                self.solve_canopy_closure_height,
                args=(n + 1, A, fG),
                bracket=(0, self.max_individual_height),
            ).root

        return z_star

    def calculate_gpp(self, cell_ppfd: NDArray, lue: NDArray) -> None:
        """Estimate the gross primary productivity.

        Not sure where to place this - need an array of LUE that matches to the

        """

        pass
