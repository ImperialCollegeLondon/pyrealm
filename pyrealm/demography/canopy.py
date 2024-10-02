"""Functionality for canopy modelling."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar  # type: ignore [import-untyped]

from pyrealm.demography.community import Community
from pyrealm.demography.crown import (
    CrownProfile,
    _validate_z_qz_args,
    calculate_relative_crown_radius_at_z,
    calculate_stem_projected_crown_area_at_z,
)


def solve_canopy_area_filling_height(
    z: float,
    stem_height: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    m: NDArray[np.float32],
    n: NDArray[np.float32],
    q_m: NDArray[np.float32],
    z_max: NDArray[np.float32],
    n_individuals: NDArray[np.float32],
    target_area: float = 0,
    validate: bool = True,
) -> NDArray[np.float32]:
    """Solver function for finding the height where a canopy occupies a given area.

    This function takes the number of individuals in each cohort along with the stem
    height and crown area and a given vertical height (:math:`z`). It then uses the
    crown shape parameters associated with each cohort to calculate the community wide
    projected crown area above that height (:math:`A_p(z)`). This is simply the sum of
    the products of the individual stem crown projected area at :math:`z` and the number
    of individuals in each cohort.

    The return value is the difference between the calculated :math:`A_p(z)` and a
    user-specified target area, This allows the function to be used with a root solver
    to find :math:`z` values that result in a given :math:`A_p(z)`. The default target
    area is zero, so the default return value will be the actual total :math:`A_p(z)`
    for the community.

    A typical use case for the target area would be to specify the area at which a given
    canopy layer closes under the perfect plasticity approximation in order to find the
    closure height.

    Args:
        z: Vertical height on the z axis.
        n_individuals: Number of individuals in each cohort
        crown_area: Crown area of each cohort
        stem_height: Stem height of each cohort
        m: Crown shape parameter ``m``` for each cohort
        n: Crown shape parameter ``n``` for each cohort
        q_m: Crown shape parameter ``q_m``` for each cohort
        z_max: Crown shape parameter ``z_m``` for each cohort
        target_area: A target projected crown area.
        validate: Boolean flag to suppress argument validation.
    """
    # Convert z to array for validation and typing
    z_arr = np.array(z)

    if validate:
        _validate_z_qz_args(
            z=z_arr,
            stem_properties=[n_individuals, crown_area, stem_height, m, n, q_m, z_max],
        )

    q_z = calculate_relative_crown_radius_at_z(
        z=z_arr, stem_height=stem_height, m=m, n=n, validate=False
    )
    # Calculate A(p) for the stems in each cohort
    A_p = calculate_stem_projected_crown_area_at_z(
        z=z_arr,
        q_z=q_z,
        stem_height=stem_height,
        crown_area=crown_area,
        q_m=q_m,
        z_max=z_max,
        validate=False,
    )

    return (A_p * n_individuals).sum() - target_area


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
        self.crown_profile: CrownProfile
        """The crown profiles of stems at layer heights."""
        self.layer_stem_leaf_area: NDArray[np.float32]
        """The leaf area of an individual stem per cohorts by layer."""
        self.layer_cohort_leaf_area: NDArray[np.float32]
        """The total leaf areas within cohorts by layer."""
        self.layer_cohort_lai: NDArray[np.float32]
        """The leaf area index for each cohort by layer."""
        self.layer_cohort_f_abs: NDArray[np.float32]
        """The fraction of light absorbed by each cohort by layer."""
        self.layer_canopy_f_abs: NDArray[np.float32]
        """The canopy-wide fraction of light absorbed by layer."""
        self.canopy_extinction_profile: NDArray[np.float32]
        """The canopy-wide light extinction profile by layer."""
        self.fapar_profile: NDArray[np.float32]
        """The canopy-wide fAPAR profile by layer."""
        self.stem_fapar: NDArray[np.float32]
        """The fAPAR for individual stems by layer."""

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
                solve_canopy_area_filling_height,
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

        # Calculate the crown profile at the layer heights
        # TODO - reimpose validation
        self.crown_profile = CrownProfile(
            stem_traits=community.stem_traits,
            stem_allometry=community.stem_allometry,
            z=self.layer_heights,
        )

        # self.canopy_projected_crown_area
        # self.canopy_projected_leaf_area

        # Partition the projected leaf area into the leaf area in each layer for each
        # stem and then scale up to the cohort leaf area in each layer.
        self.layer_stem_leaf_area = np.diff(
            self.crown_profile.projected_leaf_area, axis=0, prepend=0
        )
        self.layer_cohort_leaf_area = (
            self.layer_stem_leaf_area * community.cohort_data["n_individuals"]
        )

        # Calculate the leaf area index per layer per cohort, using the stem
        # specific leaf area index values. LAI is a value per m2, so scale back down by
        # the community area.
        self.layer_cohort_lai = (
            self.layer_cohort_leaf_area * community.stem_traits.lai
        ) / community.cell_area

        # Calculate the Beer-Lambert light extinction per layer and cohort
        self.layer_cohort_f_abs = 1 - np.exp(
            -community.stem_traits.par_ext * self.layer_cohort_lai
        )

        # Calculate the canopy wide light extinction per layer
        self.layer_canopy_f_abs = self.layer_cohort_f_abs.sum(axis=1)

        # Calculate cumulative light extinction across the canopy
        self.canopy_extinction_profile = np.cumprod(self.layer_canopy_f_abs)

        # Calculate the fraction of radiation absorbed by each layer
        # # TODO - include append=0 here to include ground or just backcalculate
        self.fapar_profile = -np.diff(
            np.cumprod(1 - self.layer_canopy_f_abs),
            prepend=1,  # append=0
        )

        # Partition the light back among the individual stems: simply weighting by per
        # cohort contribution to f_abs and divide through by the number of individuals
        self.stem_fapar = (
            self.layer_cohort_f_abs * self.fapar_profile[:, None]
        ) / self.layer_cohort_f_abs.sum(axis=1)[:, None]
