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


def fit_perfect_plasticity_approximation(
    community: Community,
    canopy_gap_fraction: float,
    max_stem_height: float,
    solver_tolerance: float,
) -> NDArray[np.float32]:
    r"""Find canopy layer heights under the PPA model.

    Finds the closure heights of the canopy layers under the perfect plasticity
    approximation by fidnding the set of heights that lead to complete closure of canopy
    layers through the canopy. The function solves the following equation for integers
    :math:`l \in (1,2,..., m)`:

    .. math::

        \sum_{s=1}^{N_s}{ A_p(z^*_l)} = l A(1 - f_G)

    The right hand side sets out the total area needed to close a given layer :math:`l`
    and all layers above it:  :math:`l` times the total community area  :math:`A` less
    any canopy gap fraction (:math:`f_G`). The left hand side then calculates the
    projected crown area for each stem :math:`s` :math:`A_p(z^*_l)_{[s]}` and sums those
    areas across all stems in the community  :math:`N_s`. The specific height
    :math:`z^*_l` is then the height at which the two terms are equal and hence solves
    the equation for layer :math:`l`.

    Args:
        community: A community instance providing plant cohort data
        canopy_gap_fraction: The canopy gap fraction
        max_stem_height: The maximum stem height in the canopy, used as an upper bound
            on finding the closure height of the topmost layer.
        solver_tolerance: The absolute tolerance used with the root solver to find the
            layer heights.
    """

    # Calculate the number of layers to contain the total community crown area
    total_community_crown_area = (
        community.stem_allometry.crown_area * community.cohort_data["n_individuals"]
    ).sum()
    crown_area_per_layer = community.cell_area * (1 - canopy_gap_fraction)
    n_layers = int(np.ceil(total_community_crown_area / crown_area_per_layer))

    # Initialise the layer heights array and then loop over the layers indices,
    # except for the final layer, which will be the partial remaining vegetation below
    # the last closed layer.
    layer_heights = np.zeros(n_layers, dtype=np.float32)
    upper_bound = max_stem_height

    for layer in np.arange(n_layers - 1):
        # Set the target area for this layer
        target_area = (layer + 1) * crown_area_per_layer

        # TODO - the solution is typically closer to the upper bound of the bracket,
        # there might be a better algorithm to find the root (#293).
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
            bracket=(0, upper_bound),
            xtol=solver_tolerance,
        )

        if not solution.converged:
            raise RuntimeError(
                "Estimation of canopy layer closure heights failed to converge."
            )

        # Store the solution and update the upper bound for the next layer down.
        layer_heights[layer] = upper_bound = solution.root

    return layer_heights[:, None]


class Canopy:
    """Calculate canopy characteristics for a plant community.

    This class generates a canopy structure for a community of trees using the
    perfect-plasticity approximation (PPA) model :cite:`purves:2008a`. In this approach,
    each individual is assumed to arrange its canopy crown area plastically to take up
    space in canopy layers and that new layers form below the canopy top as the
    available space is occupied.

    Real canopies contain canopy gaps, through process such as crown shyness. This
    is included in the model through the canopy gap fraction, which sets the proportion
    of the available space that will remain unfilled by any canopy.

    Args:
        community: A Community object that will be used to generate the canopy model.
        layer_heights: A column array of vertical heights at which to calculate canopy
            variables.
        fit_ppa: Calculate layer heights as the canopy layer closure heights under the
            PPA model.
        canopy_gap_fraction: The proportion of the available space unfilled by canopy
            (default: 0.05).
        layer_tolerance: The minimum precision used by the solver to find canopy layer
            closure heights (default: 0.001 metres)
    """

    def __init__(
        self,
        community: Community,
        layer_heights: NDArray[np.float32] | None = None,
        fit_ppa: bool = False,
        canopy_gap_fraction: float = 0,
        solver_tolerance: float = 0.001,
    ) -> None:
        # Store required init vars
        self.canopy_gap_fraction: float = canopy_gap_fraction
        """Canopy gap fraction."""
        self.solver_tolerance: float = solver_tolerance
        """Numerical tolerance for fitting the PPA model of canopy layer closure."""

        # Define class attributes
        self.max_stem_height: float
        """Maximum height of any individual in the community (m)."""
        self.n_layers: int
        """Total number of canopy layers."""
        self.n_cohorts: int
        """Total number of cohorts in the canopy."""
        self.heights: NDArray[np.float32]
        """The vertical heights at which the canopy structure is calculated."""

        self.crown_profile: CrownProfile
        """The crown profiles of the community stems at the provided layer heights."""
        self.stem_leaf_area: NDArray[np.float32]
        """The leaf area of the crown model for each cohort by layer."""
        self.cohort_lai: NDArray[np.float32]
        """The leaf area index for each cohort by layer."""
        self.cohort_f_trans: NDArray[np.float32]
        """The fraction of light transmitted by each cohort by layer."""
        self.cohort_f_abs: NDArray[np.float32]
        """The fraction of light absorbed by each cohort by layer."""
        self.f_trans: NDArray[np.float32]
        """The fraction of light transmitted by the whole community by layer."""
        self.f_abs: NDArray[np.float32]
        """The fraction of light absorbed by the whole community by layer."""
        self.transmission_profile: NDArray[np.float32]
        """The light transmission profile for the whole community by layer."""
        self.extinction_profile: NDArray[np.float32]
        """The light extinction profile for the whole community by layer."""
        self.fapar: NDArray[np.float32]
        """The fraction of absorbed radiation for the whole community by layer."""
        self.cohort_fapar: NDArray[np.float32]
        """The fraction of absorbed radiation for each cohort by layer."""
        self.stem_fapar: NDArray[np.float32]
        """The fraction of absorbed radiation for each stem by layer."""
        self.filled_community_area: float
        """The area filled by crown after accounting for the crown gap fraction."""

        # Check operating mode
        if fit_ppa ^ (layer_heights is None):
            raise ValueError("Either set fit_ppa=True or provide layer heights.")

        # Set simple attributes
        self.max_stem_height = community.stem_allometry.stem_height.max()
        self.n_cohorts = community.number_of_cohorts
        self.filled_community_area = community.cell_area * (
            1 - self.canopy_gap_fraction
        )

        # Populate layer heights
        if layer_heights is not None:
            self.heights = layer_heights
        else:
            self.heights = fit_perfect_plasticity_approximation(
                community=community,
                canopy_gap_fraction=canopy_gap_fraction,
                max_stem_height=self.max_stem_height,
                solver_tolerance=solver_tolerance,
            )

        self._calculate_canopy(community=community)

    def _calculate_canopy(self, community: Community) -> None:
        """Calculate the canopy structure.

        This private method runs the calculations needed to populate the instance
        attributes, given the layer heights provided by the user or calculated using the
        PPA model.

        Args:
            community: The Community object passed to the instance.
        """

        # Calculate the crown profile at the layer heights
        # TODO - reimpose validation
        self.crown_profile = CrownProfile(
            stem_traits=community.stem_traits,
            stem_allometry=community.stem_allometry,
            z=self.heights,
        )

        # Partition the projected leaf area into the leaf area in each layer for each
        # stem and then scale up to the cohort leaf area in each layer.
        self.stem_leaf_area = np.diff(
            self.crown_profile.projected_leaf_area, axis=0, prepend=0
        )

        # Calculate the leaf area index per layer per stem, using the stem
        # specific leaf area index values. LAI is a value per m2, so scale back down by
        # the available community area.
        self.cohort_lai = (
            self.stem_leaf_area
            * community.cohort_data["n_individuals"]
            * community.stem_traits.lai
        ) / community.cell_area  # self.filled_community_area

        # Calculate the Beer-Lambert light transmission and absorption components per
        # layer and cohort
        self.cohort_f_trans = np.exp(-community.stem_traits.par_ext * self.cohort_lai)
        self.cohort_f_abs = 1 - self.cohort_f_trans

        # Aggregate across cohorts into a layer wide transimissivity
        self.f_trans = self.cohort_f_trans.prod(axis=1)

        # Calculate the canopy wide light extinction per layer
        self.f_abs = 1 - self.f_trans

        # Calculate cumulative light transmission and extinction profiles
        self.transmission_profile = np.cumprod(self.f_trans)
        self.extinction_profile = 1 - self.transmission_profile

        # Calculate the fapar profile across cohorts and layers
        # * The first part of the equation is calculating the relative absorption of
        #   each cohort within each layer
        # * Each layer is then multiplied by fraction of the total light absorbed in the
        #   layer
        # * The resulting matrix can be multiplied by a canopy top PPFD to generate the
        #   flux absorbed within each layer for each cohort.
        self.fapar = -np.diff(self.transmission_profile, prepend=1)
        self.cohort_fapar = (
            self.cohort_f_abs / self.cohort_f_abs.sum(axis=1)[:, None]
        ) * self.fapar[:, None]
        self.stem_fapar = self.cohort_fapar / community.cohort_data["n_individuals"]
