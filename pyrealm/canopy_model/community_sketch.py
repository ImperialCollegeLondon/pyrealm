"""A very incomplete sketch of some community and demography functionality."""

import json
from dataclasses import dataclass

import numpy as np
from dataclasses_json import dataclass_json
from numpy.typing import NDArray
from scipy.optimize import root_scalar


@dataclass_json
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


@dataclass_json
@dataclass
class Cohort:
    """Contains raw imported cohort data.

    Dataclass for storing Cohort data imported from json.
    """

    pft_name: str
    dbh: float
    number_of_members: int


@dataclass_json
@dataclass
class ImportedCommunity:
    """Contains raw imported community data.

    Dataclass for storing Community data imported from json.
    """

    cell_id: int
    cohorts: list[Cohort]


@dataclass
class CohortGeometry:
    """Cohort traits.

    Includes the plant functional type of the cohort, and the geometry and mass
    calculated using the T Model. I'm not sure this all belongs in the same class
    this way. Would also like to find a better name for this class.
    """

    number_of_members: int
    pft: PlantFunctionalType
    dbh: float
    height: float
    crown_area: float
    crown_fraction: float
    mass_stem: float
    mass_fol: float
    mass_swd: float


# TODO split things into sensible modules
class PlantFunctionalTypeDeserialiser:
    """Utility class for deserialising json file to a list of plant functional types."""

    @classmethod
    def load_flora(cls, path: str) -> list[PlantFunctionalType]:
        """Loads a list of plant functional type objects from a json file.

        Uses the built-in json_dataclass functionality to deserialise a json array of
        plant functional type objects to a python list of plant functional types.
        Automatically validates type of the input using a marshmallow type schema.

        :param path: The path for a file containing a json array of Plant Functional
        Type objects.
        :return: A list of deserialised PlantFunctionalType objects.
        """
        with open(path) as file:
            pfts_json = json.load(file)
        pfts = PlantFunctionalType.schema().load(pfts_json, many=True)  # type: ignore[attr-defined] # noqa: E501
        return pfts


class CommunityDeserialiser:
    """Provides utility for deserialising json file to list of Community objects."""

    @classmethod
    def load_communities(cls, path: str) -> list[ImportedCommunity]:
        """Loads a list of community objects from a json file.

        Uses the built-in json_dataclass functionality to deserialise a json array of
        community objects to a python list of community objects. Automatically
        validates type of the input using a marshmallow type schema.
        :param path: The path for a file containing a json array of Community objects.
        :return: A deserialised list of community objects.
        """
        with open(path) as file:
            communities_json = json.load(file)
        return ImportedCommunity.schema().load(communities_json, many=True)  # type: ignore[attr-defined]  # noqa: E501


class Community:
    """Beginnings of a community class.

    This could handle multiple locations or a single location.
    """

    def __init__(
        self, flora: Flora, imported_community: ImportedCommunity, cell_area: float
    ) -> None:
        self.flora: Flora = flora
        self.cohorts: list[Cohort] = imported_community.cohorts
        self.cell_area: float = cell_area
        self.cohort_geometries: list[CohortGeometry] = list(
            map(self.calculate_cohort_geometry_using_t_model, self.cohorts)
        )

        # Things to add later

        # pft keyed dictionary of propagule density, size, energy content pft keyed
        # dictionary of propagule density, size, energy content pft keyed dictionary of
        # propagule density, size, energy content
        self.seedbank: object

        # per cohort structure of fruit mass, energy, size?
        self.fruit: object

    def calculate_cohort_geometry_using_t_model(self, cohort: Cohort) -> CohortGeometry:
        """placeholder."""

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
            crown_area * pft.rho_s * height * (1 - crown_fraction / 2) / pft.ca_ratio
        )

        return CohortGeometry(
            cohort.number_of_members,
            pft,
            cohort.dbh,
            height,
            crown_area,
            crown_fraction,
            mass_stem,
            mass_fol,
            mass_swd,
        )

    def recruit(self) -> None:
        """Add new cohorts from the seedbank."""

    pass

    def remove_empty_cohorts(self) -> None:
        """Remove cohort data for empty cohorts."""

        pass

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


def calculate_stem_canopy_factors(cohort_geometry: CohortGeometry) -> tuple[int, int]:
    """placeholder."""
    m = cohort_geometry.pft.m
    n = cohort_geometry.pft.n
    q_m = calculate_q_m(m, n)

    # Height of maximum crown radius
    z_m = cohort_geometry.height * ((n - 1) / (m * n - 1)) ** (1 / n)

    # Scaling factor to give expected Ac (crown area) at
    # z_m (height of maximum crown radius)
    r_0 = 1 / q_m * np.sqrt(cohort_geometry.crown_area / np.pi)

    return z_m, r_0


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
    cohort_geometry: CohortGeometry,
    z_resolution: float = 0.05,
) -> np.typing.NDArray:
    """Calculate the crown radius profile for a given cohort."""

    z = create_z_axis(0, cohort_geometry.height, z_resolution)

    z_m, r_0 = calculate_stem_canopy_factors(cohort_geometry)

    # calculate r(z) = r0 * q(z) for a cohort
    r_z = r_0 * calculate_relative_canopy_radius_profile(
        z, cohort_geometry.height, cohort_geometry.pft.m, cohort_geometry.pft.n
    )

    # When z > H, r_z < 0, so set radius to 0 where rz < 0
    r_z[np.where(r_z < 0)] = 0

    return r_z


def calculate_projected_canopy_area_for_individual(
    z: float, cohort_geometry: CohortGeometry
) -> float:
    """Calculate projected crown area above a given height.

    This function takes PFT specific parameters (shape parameters) and stem specific
    sizes and estimates the projected crown area above a given height $z$. Note,
    this calculation gives the canopy area for a single individual within the cohort,
    not for the cohort as a whole.
    :param z: array of heights
    :param cohort_geometry: calculated geometry of cohort
    """
    q_m = calculate_q_m(cohort_geometry.pft.m, cohort_geometry.pft.n)
    z_m, r_0 = calculate_stem_canopy_factors(cohort_geometry)

    # Calculate q(z)
    q_z = calculate_relative_canopy_radius(
        z, cohort_geometry.height, cohort_geometry.pft.m, cohort_geometry.pft.n
    )

    # Calculate A_p
    if z <= z_m:
        A_p = cohort_geometry.crown_area
    elif z > cohort_geometry.height:
        A_p = 0
    else:
        A_p = cohort_geometry.crown_area * (q_z / q_m) ** 2

    return A_p


class Canopy:
    """placeholder."""

    def __init__(self, community: Community, canopy_gap_fraction: float) -> None:
        """placeholder."""
        self.community = community
        self.cohort_geometries = community.cohort_geometries
        self.max_individual_height = max(
            cohort.height for cohort in self.cohort_geometries
        )
        self.f_g = canopy_gap_fraction

        self.canopy_layer_heights = self.calculate_canopy_layer_heights(
            community.cell_area, self.f_g
        )

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

        cohort_areas_at_z = map(
            lambda cohort: cohort.number_of_members
            * calculate_projected_canopy_area_for_individual(z, cohort),
            self.cohort_geometries,
        )

        total_projected_area_at_z = sum(cohort_areas_at_z)

        # Return the difference between the projected area and the available space
        return total_projected_area_at_z - (A * l) * (1 - fG)

    def calculate_canopy_layer_heights(self, A: float, fG: float) -> NDArray:
        """Placeholder."""
        # Calculate the number of layers
        cohort_crown_areas = map(
            lambda cohort: cohort.number_of_members * cohort.crown_area,
            self.cohort_geometries,
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

    # def calculate_canopy_layer_heights(
    #         self, community_geometry: CohortGeometries
    # ) -> None:
    #     """placeholder."""


if __name__ == "__main__":

    pfts = PlantFunctionalTypeDeserialiser.load_flora(
        "pyrealm_build_data/community/pfts.json"
    )
    flora = Flora(pfts)
    imported_communities = CommunityDeserialiser.load_communities(
        "pyrealm_build_data/community/communities.json"
    )
    for imported_community in imported_communities:
        community = Community(flora, imported_community, 32)
        canopy = Canopy(community, 2 / 32)
        print(canopy.canopy_layer_heights)
