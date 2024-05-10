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

    pft: str
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
    pft: str
    dbh: float
    height: float
    crown_area: float
    crown_fraction: float
    mass_stm: float
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
            pft = self.flora.get(cohort.pft)

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
                    cohort.pft,
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


def calculate_qm(m: float, n: float) -> float:
    """placeholder."""
    return (
        m
        * n
        * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
        * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_stem_canopy_factors(
    cohort_geometry: CohortGeometry, flora: Flora
) -> tuple[int, int]:
    """placeholder."""
    pft = flora[cohort_geometry.pft]
    m = pft.m
    n = pft.n
    qm = calculate_qm(m, n)

    # Height of maximum crown radius
    zm = cohort_geometry.height * ((n - 1) / (m * n - 1)) ** (1 / n)

    # Slope to give Ac at zm
    r0 = 1 / qm * np.sqrt(cohort_geometry.crown_area / np.pi)

    return zm, r0


class Canopy:
    """placeholder."""

    def __init__(self) -> None:
        """placeholder."""
        self.canopy_layer_heights: object

    def calculate_canopy_layer_heights(
        self, community_geometry: CommunityGeometry
    ) -> None:
        """placeholder."""
        pass
