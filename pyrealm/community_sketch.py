"""A very incomplete sketch of some community and demography functionality."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


@dataclass
class PlantFunctionalType:
    """placeholder."""

    name: str
    a_hd: float
    ca_ratio: float
    h_max: float
    lai: float
    par_ext: float
    resp_f: float
    resp_r: float
    resp_s: float
    rho_s: float
    sla: float
    tau_f: float
    tau_r: float
    yld: float
    zeta: float
    m: float
    n: float


@dataclass
class Cohort:
    """placeholder."""

    pft_name: str
    dbh: float
    number_of_members: int


class Flora(dict[str, PlantFunctionalType]):
    """Defines the flora used in a ``virtual_ecosystem`` model.

    The flora is the set of plant functional types used within a particular simulation
    and this class provides dictionary-like access to a defined set of
    :class:`~virtual_ecosystem.models.plants.functional_types.PlantFunctionalType`
    instances.

    Instances of this class should not be altered during model fitting, at least until
    the point where plant evolution is included in the modelling process.

    Args:
        pfts: A list of ``PlantFunctionalType`` instances, which must not have
            duplicated
            :attr:`~virtual_ecosystem.models.plants.functional_types.PlantFunctionalType.pft_name`
            attributes.
    """

    def __init__(self, pfts: list[PlantFunctionalType]) -> None:
        # Get the names and check there are no duplicates
        super().__init__()
        pft_names = [p.name for p in pfts]
        if len(pft_names) != len(set(pft_names)):
            msg = "Duplicated plant functional type names in creating Flora instance."
            raise ValueError(msg)

        for name, pft in zip(pft_names, pfts):
            self[name] = pft


@dataclass
class CohortGeometry:
    """placeholder."""

    number_of_members: int
    pft_name: str
    dbh: float
    height: float
    crown_area: float
    crown_fraction: float
    mass_stem: float
    mass_fol: float
    mass_swd: float


CommunityGeometry: TypeAlias = list[CohortGeometry]


class Community:
    """Beginnings of a community class.

    This could handle multiple locations or a single location.
    """

    def __init__(self, flora: Flora, cohorts: list[Cohort]) -> None:
        self.flora: Flora = flora
        self.cohorts: list[Cohort] = cohorts
        self.community_geometry: CommunityGeometry = self.calculate_geometry()

        # Things populated by methods
        self.canopy_layer_heights: NDArray[np.float32]
        self.lai_by_canopy_layer: NDArray[np.float32]

        # Things to add later

        # pft keyed dictionary of propagule density, size, energy content pft keyed
        # dictionary of propagule density, size, energy content pft keyed dictionary of
        # propagule density, size, energy content
        self.seedbank: object

        # per cohort structure of fruit mass, energy, size?
        self.fruit: object

    def recruit(self) -> None:
        """Add new cohorts from the seedbank."""

        pass

    def remove_empty_cohorts(self) -> None:
        """Remove cohort data for empty cohorts."""

        pass

    def calculate_geometry(self) -> list[CohortGeometry]:
        """placeholder."""
        results = []

        for cohort in self.cohorts:
            pft = self.flora.get(cohort.pft_name)

            # don't think this is an elegant way to do this, but putting it here for now
            if pft is None:
                raise Exception("PFT not provided in list")

            height = pft.h_max * (1 - np.exp(-pft.a_hd) * cohort.dbh / pft.h_max)

            # Crown area of tree, Equation (8) of Li ea.
            crown_area = (np.pi * pft.ca_ratio / (4 * pft.a_hd)) * cohort.dbh * height

            # Crown fraction, Equation (11) of Li ea.
            crown_fraction = height / (pft.a_hd * cohort.dbh)

            # Masses
            mass_stem = (np.pi / 8) * (cohort.dbh**2) * height * pft.rho_s
            mass_fol = crown_area * pft.lai * (1 / pft.sla)
            mass_swd = (
                crown_area
                * pft.rho_s
                * height
                * (1 - crown_fraction / 2)
                / pft.ca_ratio
            )

            results.append(
                CohortGeometry(
                    cohort.number_of_members,
                    cohort.pft_name,
                    cohort.dbh,
                    height,
                    crown_area,
                    crown_fraction,
                    mass_stem,
                    mass_fol,
                    mass_swd,
                )
            )
        return results

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
    cohort_geometry: CohortGeometry, pft: PlantFunctionalType
) -> tuple[int, int]:
    """placeholder."""
    m = pft.m
    n = pft.n
    q_m = calculate_q_m(m, n)

    # Height of maximum crown radius
    z_m = cohort_geometry.height * ((n - 1) / (m * n - 1)) ** (1 / n)

    # Scaling factor to give expected Ac (crown area) at
    # z_m (height of maximum crown radius)
    r_0 = 1 / q_m * np.sqrt(cohort_geometry.crown_area / np.pi)

    return z_m, r_0


def calculate_relative_canopy_radius_at_z(
    z: float, H: float, m: float, n: float
) -> float:
    """Calculate q(z) at a given height, z."""

    z_over_H = z / H

    return m * n * z_over_H ** (n - 1) * (1 - z_over_H**n) ** (m - 1)


calculate_relative_canopy_radius_profile = np.vectorize(
    calculate_relative_canopy_radius_at_z
)


def calculate_crown_radius_profile_for_cohort(
    cohort_geometry: CohortGeometry,
    pft: PlantFunctionalType,
    z_resolution: float = 0.05,
) -> np.typing.NDArray:
    """Calculate the crown radius profile for a given cohort."""

    max_height_padding = 1
    floating_point_correction = 0.00001

    z = np.arange(0, cohort_geometry.height + max_height_padding, z_resolution)
    z = np.sort(np.concatenate([z, cohort_geometry.height - floating_point_correction]))

    z_m, r_0 = calculate_stem_canopy_factors(cohort_geometry, pft)

    # calculate r(z) = r0 * q(z) for a cohort
    r_z = r_0 * calculate_relative_canopy_radius_profile(
        z, cohort_geometry.height, pft.m, pft.n
    )

    # When z > H, r_z < 0, so set radius to 0 where rz < 0
    r_z[np.where(r_z < 0)] = 0

    return r_z


# def calculate_projected_canopy_area_profile_for_individual(
#     z: np.ndarray, cohort_geometry: CohortGeometry, pft: PlantFunctionalType
# ) -> int:
#     """Calculate projected crown area above a given height.
#
#     This function takes PFT specific parameters (shape parameters) and stem specific
#     sizes and estimates the projected crown area above a given height $z$. The inputs
#     can either be scalars describing a single stem or arrays representing a community
#     of stems. If only a single PFT is being modelled then `m`, `n`, `qm` and `fg` can
#     be scalars with arrays `H`, `Ac` and `zm` giving the sizes of stems within that
#     PFT.
#
#     Args:
#         :param z: array of heights
#         :param cohort_geometry: calculated geometry of cohort
#         :param pft: plant functional type
#     """
#     q_m = calculate_q_m(pft.m, pft.n)
#     z_m, r_0 = calculate_stem_canopy_factors(cohort_geometry, pft)
#
#     # Calculate q(z)
#     q_z = calculate_relative_canopy_radius_profile(
#         z, cohort_geometry.height, pft.m, pft.n
#     )
#
#     # Calculate Ap given z > z_m
#     A_p = cohort_geometry.crown_area * (q_z / q_m) ** 2
#     # Set Ap = Ac where z <= z_m
#     A_p = np.where(z <= z_m, cohort_geometry.crown_area, A_p)
#     # Set Ap = 0 where z > H
#     A_p = np.where(z > cohort_geometry.height, 0, A_p)
#
#     return A_p


# def solve_canopy_closure_height(
#     z: float,
#     l: int,
#     A: float,
#     fG: float,
#     m: float,
#     n: float,
#     qm: float,
#     cohort_geometry: CohortGeometry,
#     zm: float,
# ) -> np.ndarray:
#     """Solver function for canopy closure height.
#
#     This function returns the difference between the total community projected area
#     at a height $z$ and the total available canopy space for canopy layer $l$, given
#     the community gap fraction for a given height. It is used with a root solver to
#     find canopy layer closure heights $z^*_l* for a community.
#
#     Args:
#         m, n, qm : PFT specific shape parameters
#         H, Ac, zm: stem specific sizes
#         A, l: cell area and layer index
#         fG: community gap fraction
#         :param qm:
#     """
#
#     # Calculate Ap(z)
#     Ap_z = calculate_projected_canopy_area_profile_for_individual(
#         z=z, cohort_geometry=cohort_geometry, m=m, n=n, qm=qm, zm=zm
#     )
#
#     # Return the difference between the projected area and the available space
#     return Ap_z.sum() - (A * l) * (1 - fG)
#
#
# def calculate_canopy_heights(
#     A: float,
#     fG: float,
#     m: Stems,
#     n: Stems,
#     qm: Stems,
#     pft,
#     zm: Stems,
# ):
#
#     # Calculate the number of layers
#     total_community_ca = pft.crown_area.sum()
#     n_layers = int(np.ceil(total_community_ca / (A * (1 - fG))))
#
#     # Data store for z*
#     z_star = np.zeros(n_layers)
#
#     # Loop over the layers TODO - edge case of completely filled final layer
#     for lyr in np.arange(n_layers - 1):
#         z_star[lyr] = root_scalar(
#             solve_canopy_closure_height,
#             args=(lyr + 1, A, fG, m, n, qm, pft, zm),
#             bracket=(0, pft.height.max()),
#         ).root
#
#     return z_star


class Canopy:
    """placeholder."""

    def __init__(self, community: Community) -> None:
        """placeholder."""
        self.community = community
        # self.canopy_layer_heights = self.calculate_canopy_layer_heights(
        #     community.community_geometry
        # )

    def calculate_canopy_layer_heights(
        self, community_geometry: CommunityGeometry
    ) -> None:
        """placeholder."""
