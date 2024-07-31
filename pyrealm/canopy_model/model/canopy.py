"""Functionality for canopy modelling."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from pyrealm.canopy_model.functions.jaideep_t_model_extension_functions import (
    calculate_projected_canopy_area_for_individuals,
    calculate_relative_canopy_radii,
)
from pyrealm.canopy_model.model.community import Community


class Canopy:
    """placeholder."""

    def __init__(self, community: Community, canopy_gap_fraction: float) -> None:
        """Placeholder."""
        self.max_individual_height = community.t_model_heights.max()

        self.canopy_layer_heights = self.calculate_canopy_layer_heights(
            community, canopy_gap_fraction
        )

        # TODO I would like to avoid using a map here if possible as it's really a slow
        #  for loop under the hood, but haven't thought of a way to do this and keep
        #  the code readable.
        self.A_cp_within_layer = map(
            self.calculate_total_canopy_A_cp,
            self.canopy_layer_heights,
            np.full(len(self.canopy_layer_heights), canopy_gap_fraction),
            np.full(len(self.canopy_layer_heights), community),
        )

    def calculate_canopy_layer_heights(
        self, community: Community, fG: float
    ) -> NDArray:
        """Placeholder."""
        # Calculate the number of layers
        cohort_crown_areas = (
            community.cohort_number_of_individuals * community.t_model_crown_areas
        )
        total_community_crown_area = cohort_crown_areas.sum()
        number_of_layers = int(
            np.ceil(total_community_crown_area / (community.cell_area * (1 - fG)))
        )

        # Data store for z*
        z_star = np.zeros(number_of_layers)

        # Loop over the layers TODO - edge case of completely filled final layer
        for n in np.arange(number_of_layers - 1):
            z_star[n] = root_scalar(
                self.solve_canopy_closure_height,
                args=(community, n + 1, community.cell_area, fG),
                bracket=(0, self.max_individual_height),
            ).root

        return z_star

    def calculate_total_canopy_A_cp(
        self, z: float, f_g: float, community: Community
    ) -> float:
        """Calculate total leaf area at a given height.

        :param f_g:
        :param community:
        :param z: Height above ground.
        :return: Total leaf area in the canopy at a given height.
        """
        A_cp_for_individuals = self.calculate_projected_leaf_area_for_individuals(
            z, f_g, community
        )

        A_cp_for_cohorts = A_cp_for_individuals * community.cohort_number_of_individuals

        return A_cp_for_cohorts.sum()

    @classmethod
    def solve_canopy_closure_height(
        cls,
        z: float,
        community: Community,
        l: int,
        A: float,
        fG: float,
    ) -> float:
        """Solver function for canopy closure height.

        This function returns the difference between the total community projected area
        at a height $z$ and the total available canopy space for canopy layer $l$, given
        the community gap fraction for a given height. It is used with a root solver to
        find canopy layer closure heights $z^*_l* for a community.
        :param community:
        :param fG: community gap fraction
        :param A: community area
        :param l: layer index
        :param z: height
        """

        community_projected_area_at_z = cls.calculate_community_projected_area_at_z(
            community, z
        )

        # Return the difference between the projected area and the available space
        return community_projected_area_at_z - (A * l) * (1 - fG)

    @classmethod
    def calculate_community_projected_area_at_z(
        cls, community: Community, z: float
    ) -> float:
        """Calculate the total area of community stems."""
        projected_canopy_area_for_individuals = (
            calculate_projected_canopy_area_for_individuals(
                z,
                community.t_model_heights,
                community.t_model_crown_areas,
                community.pft_m_values,
                community.pft_n_values,
                community.canopy_factor_q_m_values,
                community.canopy_factor_z_m_values,
            )
        )

        cohort_areas_at_z = (
            community.cohort_number_of_individuals
            * projected_canopy_area_for_individuals
        )

        return sum(cohort_areas_at_z)

    def calculate_gpp(self, cell_ppfd: NDArray, lue: NDArray) -> None:
        """Estimate the gross primary productivity.

        Not sure where to place this - need an array of LUE that matches to the

        """

        pass

    @classmethod
    def calculate_projected_leaf_area_for_individuals(
        cls, z: float, f_g: float, community: Community
    ) -> float:
        """Calculate projected crown area above a given height.

        Calculation applies to an individual within a cohort.This function takes PFT
        specific parameters (shape parameters) and stem specific sizes and estimates
        the projected crown area above a given height $z$. The inputs can either be
        scalars describing a single stem or arrays representing a community of stems.
        If only a single PFT is being modelled then `m`, `n`, `qm` and `fg` can be
        scalars with arrays `H`, `Ac` and `zm` giving the sizes of stems within that
        PFT.

        :param community:
        :param f_g: Crown gap fraction.
        :param z: Height from ground.
        """

        # Calculate q(z)
        q_z = calculate_relative_canopy_radii(
            z, community.t_model_heights, community.pft_m_values, community.pft_n_values
        )

        # Calculate Ac terms
        A_c_terms = (
            community.t_model_crown_areas
            * (q_z / community.canopy_factor_q_m_values) ** 2
        )

        # Set Acp either side of zm
        A_cp = np.where(
            z <= community.canopy_factor_z_m_values,
            community.t_model_crown_areas - A_c_terms * f_g,
            A_c_terms * (1 - f_g),
        )
        # Set Ap = 0 where z > H
        A_cp = np.where(z > community.t_model_heights, 0, A_cp)
        return A_cp
