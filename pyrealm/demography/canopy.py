"""Functionality for canopy modelling."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar  # type: ignore [import-untyped]

from pyrealm.demography.community import Community
from pyrealm.demography.crown import (
    calculate_relative_crown_radius_at_z,
    calculate_stem_projected_crown_area_at_z,
    calculate_stem_projected_leaf_area_at_z,
    solve_community_projected_canopy_area,
)


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
        self.canopy_gap_fraction: float = canopy_gap_fraction
        """Canopy gap fraction."""
        self.layer_tolerance: float = layer_tolerance
        """Numerical tolerance for solving canopy layer closure."""
        self.total_community_crown_area: float
        """Total crown area across individuals in the community (metres 2)."""
        self.max_stem_height: float
        """Maximum height of any individual in the community (metres)."""
        self.crown_area_per_layer: float
        """Total crown area permitted in a single canopy layer, given the available
        cell area of the community and its canopy gap fraction."""
        self.n_layers: int
        """Total number of canopy layers."""
        self.n_cohorts: int
        """Total number of cohorts in the canopy."""
        self.layer_heights: NDArray[np.float32]
        """Column vector of the heights of canopy layers."""
        self.stem_relative_radius: NDArray[np.float32]
        """Relative radius values of stems at canopy layer heights."""
        self.stem_crown_area: NDArray[np.float32]
        """Stem projected crown area at canopy layer heights."""
        self.stem_leaf_area: NDArray[np.float32]
        """Stem projected leaf area at canopy layer heights."""

        self._calculate_canopy(community=community)

    def _calculate_canopy(self, community: Community) -> None:
        """Calculate the canopy structure.

        This private method runs the calculations needed to populate the instance
        attributes.

        Args:
            community: The Community object passed to the instance.
        """

        # Calculate community wide properties: total crown area, maximum height, crown
        # area required to fill a layer and total number of canopy layers
        self.total_community_crown_area = (
            community.stem_allometry.crown_area * community.cohort_data["n_individuals"]
        ).sum()

        self.max_stem_height = community.stem_allometry.stem_height.max()

        self.crown_area_per_layer = community.cell_area * (1 - self.canopy_gap_fraction)

        self.n_layers = int(
            np.ceil(self.total_community_crown_area / self.crown_area_per_layer)
        )
        self.n_cohorts = community.number_of_cohorts

        # Find the closure heights of the canopy layers under the perfect plasticity
        # approximation by solving Ac(z) - L_n = 0 across the community where L is the
        # total cumulative crown area in layer n and above, discounted by the canopy gap
        # fraction.

        self.layer_heights = np.zeros((self.n_layers, 1), dtype=np.float32)

        # Loop over the layers except for the final layer, which will be the partial
        # remaining vegetation below the last closed layer.
        starting_guess = self.max_stem_height
        for layer in np.arange(self.n_layers - 1):
            target_area = (layer + 1) * self.crown_area_per_layer

            # TODO - the solution here is typically closer to the upper bracket, might
            # be a better algorithm to find the root (#293).
            solution = root_scalar(
                solve_community_projected_canopy_area,
                args=(
                    community.stem_allometry.stem_height,
                    community.stem_allometry.crown_area,
                    community.stem_traits.m,
                    community.stem_traits.n,
                    community.stem_traits.q_m,
                    community.stem_allometry.crown_z_max,
                    community.cohort_data["n_individuals"],
                    target_area,
                    False,  # validate
                ),
                bracket=(0, starting_guess),
                xtol=self.layer_tolerance,
            )

            if not solution.converged:
                raise RuntimeError(
                    "Estimation of canopy layer closure heights failed to converge."
                )

            self.layer_heights[layer] = starting_guess = solution.root

        # Find relative canopy radius at the layer heights
        # NOTE - here and in the calls below, validate=False is enforced because the
        # Community class structures and code should guarantee valid inputs and so
        # turning off the validation internally should simply speed up the code.
        self.stem_relative_radius = calculate_relative_crown_radius_at_z(
            z=self.layer_heights,
            stem_height=community.stem_allometry.stem_height,
            m=community.stem_traits.m,
            n=community.stem_traits.n,
            validate=False,
        )

        # Calculate projected crown area of a cohort stem at canopy closure heights.
        self.stem_crown_area = calculate_stem_projected_crown_area_at_z(
            z=self.layer_heights,
            q_z=self.stem_relative_radius,
            crown_area=community.stem_allometry.crown_area,
            stem_height=community.stem_allometry.stem_height,
            q_m=community.stem_traits.q_m,
            z_max=community.stem_allometry.crown_z_max,
            validate=False,
        )

        # Find the projected leaf area of a cohort stem at canopy closure heights.
        self.stem_leaf_area = calculate_stem_projected_leaf_area_at_z(
            z=self.layer_heights,
            q_z=self.stem_relative_radius,
            crown_area=community.stem_allometry.crown_area,
            stem_height=community.stem_allometry.stem_height,
            f_g=community.stem_traits.f_g,
            q_m=community.stem_traits.q_m,
            z_max=community.stem_allometry.crown_z_max,
            validate=False,
        )
