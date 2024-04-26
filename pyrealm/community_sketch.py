"""A very incomplete sketch of some community and demography functionality."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray


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

    def __init__(self, flora: Flora, inventory: pd.DataFrame) -> None:
        self.flora: Flora = flora

        # TODO - validate the inventory data (field names and values)
        self.inventory = inventory

        # Things populated by methods
        self.canopy_layer_heights: NDArray[np.float32]
        self.lai_by_canopy_layer: NDArray[np.float32]

        self.calculate_geometry()
        self.calculate_canopy()

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

    def calculate_canopy(self) -> None:
        """Uses the canopy model to get the canopy structure.

        If the class handles multiple communities, this needs to loop over each location
        and it is probably easier not to.

        Sets self.layer_heights and a per cohort area of leaf area within each layer.
        """

        # TODO:
        # - implement canopy calculation
        pass

    def calculate_gpp(self, cell_ppfd: NDArray, lue: NDArray) -> None:
        """Estimate the gross primary productivity.

        Not sure where to place this - need an array of LUE that matches to the

        """

        pass
