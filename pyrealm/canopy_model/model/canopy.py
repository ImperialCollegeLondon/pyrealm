"""Functionality for canopy modelling."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from pyrealm.canopy_model.model.cohort import Cohort
from pyrealm.canopy_model.model.community import Community
from pyrealm.canopy_model.model.jaideep_t_model_extension import (
    calculate_projected_canopy_area_for_individual,
    calculate_relative_canopy_radius,
)


class Canopy:
    """placeholder."""

    def __init__(self, community: Community, canopy_gap_fraction: float) -> None:
        """placeholder."""
        self.f_g = canopy_gap_fraction

        self.max_individual_height = community.t_model_heights.max()

        self.canopy_layer_heights = self.calculate_canopy_layer_heights(community, self.f_g)

        self.A_cp_within_layer = map(
            self.calculate_total_canopy_A_cp, self.canopy_layer_heights
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

    def calculate_canopy_layer_heights(self, community: Community, fG: float) -> NDArray:
        """Placeholder."""
        # Calculate the number of layers
        cohort_crown_areas = (community.cohort_number_of_individuals
                              * community.t_model_crown_areas)
        total_community_crown_area = cohort_crown_areas.sum()
        number_of_layers = int(np.ceil(total_community_crown_area / (community.cell_area * (1 - fG))))

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

    @classmethod
    def calculate_projected_leaf_area_for_individual(
        cls, z: float, f_g: float, cohort: Cohort
    ) -> float:
        """Calculate projected crown area above a given height.

        Calculation applies to an individual within a cohort.This function takes PFT
        specific parameters (shape parameters) and stem specific sizes and estimates
        the projected crown area above a given height $z$. The inputs can either be
        scalars describing a single stem or arrays representing a community of stems.
        If only a single PFT is being modelled then `m`, `n`, `qm` and `fg` can be
        scalars with arrays `H`, `Ac` and `zm` giving the sizes of stems within that
        PFT.

        :param f_g: Crown gap fraction.
        :param z: Height from ground.
        :param cohort: a cohort consisting of plants with the same diameter at
        breast height and plant functional type.

        """

        # Calculate q(z)
        q_z = calculate_relative_canopy_radius(
            z, cohort.t_model_geometry.height, cohort.pft.m, cohort.pft.n
        )

        # Calculate Ac term
        Ac_term = (
            cohort.t_model_geometry.crown_area * (q_z / cohort.canopy_factors.q_m) ** 2
        )

        if z <= cohort.canopy_factors.z_m:
            A_cp = cohort.t_model_geometry.crown_area - Ac_term * f_g
        elif z > cohort.t_model_geometry.height:
            A_cp = 0
        else:
            A_cp = Ac_term * (1 - f_g)

        return A_cp

    def calculate_total_canopy_A_cp(self, z: float) -> float:
        """Calculate total leaf area at a given height.

        :param z: Height above ground.
        :return: Total leaf area in the canopy at a given height.
        """
        A_cp_for_cohorts = list(
            map(
                lambda cohort: cohort.number_of_members
                * self.calculate_projected_leaf_area_for_individual(z, 0.05, cohort),
                self.cohorts,
            )
        )
        return sum(A_cp_for_cohorts)
