"""A very incomplete sketch of some community and demography functionality."""

from importlib import resources

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import root_scalar


class Flora:
    """Beginnings of a container class for PFTs."""

    def __init__(self, pft_data: pd.DataFrame):
        # TODO:
        # - validate the incoming data against the expectations of the T model
        #   parameterisation.
        # - Where are those defaults now defined? Just use TModelTraits as a defaults
        #   argument constructor? Provides expected fields and defaults.
        # - support automatic default values?
        # - add canopy shape parameters m and n and internally calculate derived
        #   constant q_m for canopy calculations.

        self.pft_data = pft_data
        self.pft_names = set(pft_data["pft"])


class Community:
    """Beginnings of a community class.

    This could handle multiple locations or a single location.
    """

    def __init__(self, flora: Flora, inventory: pd.DataFrame, cell_area: float) -> None:
        self.flora: Flora = flora
        self.cell_area: float = cell_area

        # TODO - validate the inventory data (field names and values)
        self.inventory = inventory

        # Things populated by methods

        self.calculate_geometry()

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

    def calculate_geometry(self) -> None:
        """Calculates the geometry and mass data for each cohort."""

        # NOTE - Don't know if this makes a difference, trying to avoid many
        #        getattr(self.data, 'xyz') calls by using a local reference.
        inventory = self.inventory

        # Broadcast the PFT trait values onto the matching rows in the inventory data
        # TODO: handle NAs following merge or just rely on robust initial validation
        #       that all inventory pfts exist in flora.pft_names
        # NOTE: is it better to avoid repeated merge operations by making the inventory
        #       include the trait data at __init__ and when cohorts are added. Cost of
        #       increasing data storage and repeated data, but probably faster.
        traits = inventory[["pft"]].merge(self.flora.pft_data, how="left")

        # Height of tree from diameter, Equation (4) of Li ea.
        inventory["height"] = traits["h_max"] * (
            1 - np.exp(-traits["a_hd"] * inventory["dbh"] / traits["h_max"])
        )

        # Crown area of tree, Equation (8) of Li ea.
        inventory["crown_area"] = (
            ((np.pi * traits["ca_ratio"]) / (4 * traits["a_hd"]))
            * inventory["dbh"]
            * inventory["height"]
        )

        # Crown fraction, Equation (11) of Li ea.
        inventory["crown_fraction"] = inventory["height"] / (
            traits["a_hd"] * inventory["dbh"]
        )

        # Masses
        inventory["mass_stm"] = (
            (np.pi / 8)
            * (inventory["dbh"] ** 2)
            * inventory["height"]
            * traits["rho_s"]
        )
        inventory["mass_fol"] = (
            inventory["crown_area"] * traits["lai"] * (1 / traits["sla"])
        )
        inventory["mass_swd"] = (
            inventory["crown_area"]
            * traits["rho_s"]
            * inventory["height"]
            * (1 - inventory["crown_fraction"] / 2)
            / traits["ca_ratio"]
        )

        # TODO: Use canopy constants (m, n) to calculate

        # Update stored inventory with added geometry
        self.inventory = inventory

    def calculate_gpp(self, cell_ppfd: NDArray, lue: NDArray) -> None:
        """Estimate the gross primary productivity.

        Not sure where to place this - need an array of LUE that matches to the

        """

        pass


class Canopy:
    """Class representing the canopy for a given community.

    Class representing the canopy for a given community, including the heights of
    the canopy layers.
    """

    def __init__(self, community: Community):
        self.inventory = community.inventory
        self.canopy_layer_heights: NDArray[np.float32] = self.__calculate_canopy()

    def __calculate_canopy(self) -> NDArray:
        self.__calculate_qm_for_inventory()
        self.__calculate_stem_canopy_factors_for_inventory()
        return self.__calculate_canopy_layer_heights(community.cell_area, 2 / 32)

    def __calculate_qm_for_inventory(self) -> None:
        inventory = self.inventory
        inventory["q_m"] = (
            inventory["m"]
            * inventory["n"]
            * ((inventory["n"] - 1) / (inventory["m"] * inventory["n"] - 1))
            ** (1 - 1 / inventory["n"])
            * (
                ((inventory["m"] - 1) * inventory["n"])
                / (inventory["m"] * inventory["n"] - 1)
            )
            ** (inventory["m"] - 1)
        )
        self.inventory = inventory

    def __calculate_stem_canopy_factors_for_inventory(self) -> None:
        """Calculates stem canopy factors.

        Calculates the stem canopy factors, r_0 and z_m for every cohort in the
        inventory and stores them in the inventory dataframe. :return:
        """
        inventory = self.inventory
        # Height of maximum crown radius
        inventory["z_m"] = inventory["height"] * (
            (inventory["n"] - 1) / (inventory["m"] * inventory["n"] - 1)
        ) ** (1 / inventory["n"])

        # Slope to give Ac at zm
        inventory["r_0"] = (
            1 / inventory["q_m"] * np.sqrt(inventory["crown_area"] / np.pi)
        )
        self.inventory = inventory

    def __calculate_relative_canopy_radius_at_z(self, z: float) -> pd.Series[float]:
        """Calculate q(z)."""

        inventory = self.inventory

        z_over_H = z / inventory["height"]

        return (
            inventory["m"]
            * inventory["n"]
            * z_over_H ** (inventory["n"] - 1)
            * (1 - z_over_H ** inventory["n"]) ** (inventory["m"] - 1)
        )

    def __calculate_projected_area_at_z(self, z: float) -> pd.Series[float]:
        """Calculate projected crown area above a given height.

        This function takes PFT specific parameters (shape parameters) and stem
        specific sizes and estimates the projected crown area above a given height
        $z$. The inputs can either be scalars describing a single stem or arrays
        representing a community of stems. If only a single PFT is being modelled
        then `m`, `n`, `qm` and `fg` can be scalars with arrays `H`, `Ac` and `zm`
        giving the sizes of stems within that PFT.

        Args:
            z: Canopy height
            m, n, qm : PFT specific shape parameters
            pft, qm, zm: stem data

        """

        inventory = self.inventory

        # Calculate q(z)
        q_z = self.__calculate_relative_canopy_radius_at_z(z)

        # Calculate Ap for an individual in the cohort
        # Calculate Ap given z > zm
        Ap = inventory["crown_area"] * (q_z / inventory["q_m"]) ** 2
        # Set Ap = Ac where z <= zm
        Ap.where(z <= inventory["z_m"], inventory["crown_area"])
        # Set Ap = 0 where z > H
        Ap.where(z > inventory["height"], 0)

        # think this will be a panda series but need to check...
        return Ap

    def solve_canopy_closure_height(
        self, z: float, layer_index: int, A: float, fG: float
    ) -> np.ndarray:
        """Solver function for canopy closure height.

        This function returns the difference between the total community projected area
        at a height $z$ and the total available canopy space for canopy layer $l$, given
        the community gap fraction for a given height. It is used with a root solver to
        find canopy layer closure heights $z^*_l* for a community.

        Args:
            z: height
            A: cell area
            layer_index:  layer index
            fG: community gap fraction
        """

        # Calculate Ap(z) for an individual in the cohort
        Ap_z = self.__calculate_projected_area_at_z(z=z)

        cohort_Ap_z = self.inventory["number_of_members"] * Ap_z

        # Return the difference between the projected area and the available space
        return cohort_Ap_z.sum() - (A * layer_index) * (1 - fG)

    def __calculate_canopy_layer_heights(self, A: float, fG: float) -> NDArray:
        """Calculate the heights of the canopy layers.

        Calculate the heights of the canopy layers.
        :param A: Cell area
        :param fG:
        :return:
        """
        inventory = self.inventory

        # Calculate the number of layers
        inventory["cohort_crown_areas"] = (
            inventory["crown_area"] * inventory["number_of_members"]
        )
        total_community_ca = inventory["cohort_crown_areas"].sum()
        n_layers = int(np.ceil(total_community_ca / (A * (1 - fG))))

        # Data store for z*
        z_star = np.zeros(n_layers)

        # Loop over the layers TODO - edge case of completely filled final layer
        for lyr in np.arange(n_layers - 1):
            z_star[lyr] = root_scalar(
                self.solve_canopy_closure_height,
                args=(lyr + 1, A, fG),
                bracket=(0, inventory["height"].max()),
            ).root

        return z_star


if __name__ == "__main__":
    # Load the data from the build data
    dpath = resources.files("pyrealm_build_data.community")
    pft_data = pd.read_csv("pyrealm_build_data/community/pfts.csv")
    inventory = pd.read_csv("pyrealm_build_data/community/community.csv")

    # Create the objects
    flora = Flora(pft_data=pft_data)
    community = Community(flora=flora, inventory=inventory, cell_area=32)

    # A thing has been calculated
    print(community.inventory["height"])

    canopy = Canopy(community)

    print(canopy.canopy_layer_heights)
