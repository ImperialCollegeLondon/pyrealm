"""Functionality for canopy modelling."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from pyrealm.demography.canopy_functions import (
    calculate_relative_canopy_radius_at_z,
    calculate_stem_projected_crown_area_at_z,
    solve_community_projected_canopy_area,
)
from pyrealm.demography.community import Community


class Canopy:
    """A class containing attributes of a canopy, including the structure."""

    def __init__(self, community: Community, canopy_gap_fraction: float) -> None:
        # Calculate community wide properties: total crown area and maximum height
        self.total_community_crown_area: float = (
            community.cohort_data["crown_area"] * community.cohort_data["n_individuals"]
        ).sum()
        """Total crown area across individuals in the community (metres 2)."""

        self.max_stem_height: float = community.cohort_data["stem_height"].max()
        """Maximum height of any individual in the community (metres)."""

        self.crown_area_per_layer: float = community.cell_area * (
            1 - canopy_gap_fraction
        )
        """Total crown area permitted in a single canopy layer, given the available
        cell area of the community and its canopy gap fraction."""

        self.n_layers: int = int(
            np.ceil(self.total_community_crown_area / self.crown_area_per_layer)
        )
        """Total number of canopy layers needed to contain the total crown area."""

        # Find the closure heights of the canopy layers under the perfect plasticity
        # approximation by solving Ac(z) - L_n = 0 across the community where L is the
        # total cumulative crown area in layer n and above, discounted by the canopy gap
        # fraction.

        self.layer_closure_heights: NDArray[np.float32] = np.full(
            (self.n_layers), np.nan
        )

        # Loop over the layers TODO - check edge case of completely filled final layer
        for layer in np.arange(self.n_layers):
            target_area = (layer + 1) * community.cell_area * (1 - canopy_gap_fraction)

            self.layer_closure_heights[layer] = root_scalar(
                solve_community_projected_canopy_area,
                args=(
                    community.cohort_data["n_individuals"],  # n_individuals
                    community.cohort_data["crown_area"],  # crown_area
                    community.cohort_data["stem_height"],  # stem_height
                    community.cohort_data["m"],  # m
                    community.cohort_data["n"],  # n
                    community.cohort_data["canopy_q_m"],  # q_m
                    community.cohort_data["canopy_z_max"],  # z_m
                    target_area,  # target_area
                    False,  # validate
                ),
                bracket=(0, self.max_stem_height),
            ).root

        # Find the stem projected canopy area and relative canopy radius at canopy
        # closure heights
        self.stem_relative_radius = calculate_relative_canopy_radius_at_z(
            z=self.layer_closure_heights[:, None],
            stem_height=community.cohort_data["stem_height"].to_numpy(),
            m=community.cohort_data["m"].to_numpy(),
            n=community.cohort_data["n"].to_numpy(),
            validate=False,
        )

        self.stem_crown_area = calculate_stem_projected_crown_area_at_z(
            z=self.layer_closure_heights[:, None],
            q_z=self.stem_relative_radius,
            crown_area=community.cohort_data["crown_area"].to_numpy(),
            stem_height=community.cohort_data["stem_height"].to_numpy(),
            q_m=community.cohort_data["canopy_q_m"].to_numpy(),
            z_max=community.cohort_data["canopy_z_max"].to_numpy(),
            validate=False,
        )

        # Find the stem projected leaf area at canopy closure heights.

        # # TODO there may be a more efficient solution here that does not use a loop.
        # self.A_cp_within_layer = map(
        #     calculate_total_canopy_A_cp,
        #     self.canopy_layer_heights,
        #     np.full(self.number_of_canopy_layers, canopy_gap_fraction),
        #     np.full(self.number_of_canopy_layers, community),
        # )
