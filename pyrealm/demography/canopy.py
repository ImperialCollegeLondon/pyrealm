"""Functionality for canopy modelling."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar  # type: ignore [import-untyped]

from pyrealm.demography.canopy_functions import (
    calculate_relative_canopy_radius_at_z,
    calculate_stem_projected_crown_area_at_z,
    calculate_stem_projected_leaf_area_at_z,
    solve_community_projected_canopy_area,
)
from pyrealm.demography.community import Community


class Canopy:
    """Model of the canopy for a plant community.

    This class generates a canopy structure for a community of trees using the
    perfect-plasticity approximation model :cite:`purves:2008a`. In this approach, each
    individual is assumed to arrange its canopy crown area plastically to take up space
    in canopy layers and that new layers form below the canopy top as the available
    space is occupied.

    Real canopies contain canopy gaps, through process such as crown shyness. This
    is included in the model through the canopy gap fraction, which sets the proportion
    of the available space that will remain unfilled by any canopy.

    Args:
        community: A Community object that will be used to generate the canopy model.
        canopy_gap_fraction: The proportion of the available space unfilled by canopy
            (default: 0.05).
        layer_tolerance: The minimum precision used by the solver to find canopy layer
            closure heights (default: 0.001 metres)
    """

    def __init__(
        self,
        community: Community,
        canopy_gap_fraction: float = 0.05,
        layer_tolerance: float = 0.001,
    ) -> None:
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
        """Total number of canopy layers."""

        self.n_cohorts: int = community.number_of_cohorts
        """Total number of cohorts in the canopy."""

        # Find the closure heights of the canopy layers under the perfect plasticity
        # approximation by solving Ac(z) - L_n = 0 across the community where L is the
        # total cumulative crown area in layer n and above, discounted by the canopy gap
        # fraction.

        self.layer_heights: NDArray[np.float32] = np.zeros(
            (self.n_layers, 1), dtype=np.float32
        )
        """Column vector of the heights of canopy layers."""

        # Loop over the layers except for the final layer, which will be the partial
        # remaining vegetation below the last closed layer.
        starting_guess = self.max_stem_height
        for layer in np.arange(self.n_layers - 1):
            target_area = (layer + 1) * community.cell_area * (1 - canopy_gap_fraction)

            # TODO - the solution here is predictably closer to the upper bracket, might
            # be a better algorithm to find the root.
            solution = root_scalar(
                solve_community_projected_canopy_area,
                args=(
                    community.cohort_data["stem_height"],
                    community.cohort_data["crown_area"],
                    community.cohort_data["m"],
                    community.cohort_data["n"],
                    community.cohort_data["q_m"],
                    community.cohort_data["canopy_z_max"],
                    community.cohort_data["n_individuals"],
                    target_area,
                    False,  # validate
                ),
                bracket=(0, starting_guess),
                xtol=layer_tolerance,
            )

            if not solution.converged:
                raise RuntimeError(
                    "Estimation of canopy layer closure heights failed to converge."
                )

            self.layer_heights[layer] = starting_guess = solution.root

        # Find relative canopy radius at the layer heights
        self.stem_relative_radius: NDArray[np.float32] = (
            calculate_relative_canopy_radius_at_z(
                z=self.layer_heights,
                stem_height=community.cohort_data["stem_height"],
                m=community.cohort_data["m"],
                n=community.cohort_data["n"],
                validate=False,
            )
        )
        """Relative radius values of stems at canopy layer heights."""

        self.stem_crown_area: NDArray[np.float32] = (
            calculate_stem_projected_crown_area_at_z(
                z=self.layer_heights,
                q_z=self.stem_relative_radius,
                crown_area=community.cohort_data["crown_area"],
                stem_height=community.cohort_data["stem_height"],
                q_m=community.cohort_data["q_m"],
                z_max=community.cohort_data["canopy_z_max"],
                validate=False,
            )
        )
        """Stem projected crown area at canopy layer heights."""

        # Find the stem projected leaf area at canopy closure heights.
        self.stem_leaf_area: NDArray[np.float32] = (
            calculate_stem_projected_leaf_area_at_z(
                z=self.layer_heights,
                q_z=self.stem_relative_radius,
                crown_area=community.cohort_data["crown_area"],
                stem_height=community.cohort_data["stem_height"],
                f_g=community.cohort_data["f_g"],
                q_m=community.cohort_data["q_m"],
                z_max=community.cohort_data["canopy_z_max"],
                validate=False,
            )
        )
        """Stem projected leaf area at canopy layer heights."""
