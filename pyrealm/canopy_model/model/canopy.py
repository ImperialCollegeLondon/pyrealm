"""Functionality for canopy modelling."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from pyrealm.canopy_model.model.cohort import TModelGeometry
from pyrealm.canopy_model.model.community import Community
from pyrealm.canopy_model.model.flora import PlantFunctionalType


@dataclass
class CanopyFactors:
    """Canopy factors calculated from Jaideep's extension to the T Model."""

    q_m: float
    z_m: float
    r_0: float


class Canopy:
    """placeholder."""

    def __init__(self, community: Community, canopy_gap_fraction: float) -> None:
        """placeholder."""
        self.f_g = canopy_gap_fraction
        self.cohorts = community.cohorts

        self.canopy_factors: list[CanopyFactors] = list(
            map(
                lambda cohort: calculate_stem_canopy_factors(
                    cohort.pft, cohort.t_model_geometry
                ),
                self.cohorts,
            )
        )

        self.max_individual_height = max(
            cohort.t_model_geometry.height for cohort in self.cohorts
        )

        self.canopy_layer_heights = self.calculate_canopy_layer_heights(
            community.cell_area, self.f_g
        )

    def calculate_community_projected_area_at_z(self, z: float) -> float:
        """Calculate the total area of community stems."""
        cohort_areas_at_z = map(
            lambda cohort, canopy_f: cohort.number_of_members
            * calculate_projected_canopy_area_for_individual(
                z, cohort.pft, cohort.t_model_geometry, canopy_f
            ),
            self.cohorts,
            self.canopy_factors,
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


def calculate_q_m(m: float, n: float) -> float:
    """placeholder."""
    return (
        m
        * n
        * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
        * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_stem_canopy_factors(
    pft: PlantFunctionalType, t_model_geometry: TModelGeometry
) -> CanopyFactors:
    """Calculate stem canopy factors from Jaideep's extension to the T Model."""
    m = pft.m
    n = pft.n
    q_m = calculate_q_m(m, n)

    # Height of maximum crown radius
    z_m = t_model_geometry.height * ((n - 1) / (m * n - 1)) ** (1 / n)

    # Scaling factor to give expected Ac (crown area) at
    # z_m (height of maximum crown radius)
    r_0 = 1 / q_m * np.sqrt(t_model_geometry.crown_area / np.pi)

    return CanopyFactors(q_m, z_m, r_0)


def calculate_relative_canopy_radius(z: float, H: float, m: float, n: float) -> float:
    """Calculate q(z) at a given height, z."""

    z_over_H = z / H

    return m * n * z_over_H ** (n - 1) * (1 - z_over_H**n) ** (m - 1)


calculate_relative_canopy_radius_profile = np.vectorize(
    calculate_relative_canopy_radius
)


def create_z_axis(
    z_min: float, z_max: float, resolution: float = 0.05
) -> np.typing.NDArray:
    """Provides a z axis in the form of a numpy array.

    :param z_min: start of the axis
    :param z_max: end of the axis
    :param resolution: resolution of the axis
    :return: a z axis from z_min, to z_max, in increments of the resolution
    """
    max_height_padding = 1
    floating_point_correction = 0.00001

    z = np.arange(z_min, z_max + max_height_padding, resolution)
    z = np.sort(np.concatenate([z, z_max - floating_point_correction]))

    return z


def calculate_crown_radius_profile_for_cohort(
    pft: PlantFunctionalType,
    t_model_geometry: TModelGeometry,
    z_resolution: float = 0.05,
) -> np.typing.NDArray:
    """Calculate the crown radius profile for a given cohort."""

    z = create_z_axis(0, t_model_geometry.height, z_resolution)

    canopy_factors = calculate_stem_canopy_factors(pft, t_model_geometry)

    # calculate r(z) = r0 * q(z) for a cohort
    r_z = canopy_factors.r_0 * calculate_relative_canopy_radius_profile(
        z, t_model_geometry.height, pft.m, pft.n
    )

    # When z > H, r_z < 0, so set radius to 0 where rz < 0
    r_z[np.where(r_z < 0)] = 0

    return r_z


def calculate_projected_canopy_area_for_individual(
    z: float,
    pft: PlantFunctionalType,
    t_model_geometry: TModelGeometry,
    canopy_factors: CanopyFactors,
) -> float:
    """Calculate projected crown area above a given height.

    This function takes PFT specific parameters (shape parameters) and stem specific
    sizes and estimates the projected crown area above a given height $z$. Note,
    this calculation gives the canopy area for a single individual within the cohort,
    not for the cohort as a whole.
    :param canopy_factors:
    :param pft:
    :param z_m: stem canopy factor from Jaideep’s extension of the T Model.
    :param q_m: stem canopy factor from Jaideep’s extension of the T Model.
    :param z: height on the z axis.
    :param t_model_geometry: calculated geometry of cohort using T model.
    """

    # Calculate q(z)
    q_z = calculate_relative_canopy_radius(z, t_model_geometry.height, pft.m, pft.n)

    # Calculate A_p
    if z <= canopy_factors.z_m:
        A_p = t_model_geometry.crown_area
    elif z > t_model_geometry.height:
        A_p = 0
    else:
        A_p = t_model_geometry.crown_area * (q_z / canopy_factors.q_m) ** 2

    return A_p
