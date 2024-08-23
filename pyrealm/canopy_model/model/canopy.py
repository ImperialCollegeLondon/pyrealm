"""Functionality for canopy modelling."""

import numpy as np

from pyrealm.canopy_model.functions.canopy_functions import (
    calculate_canopy_layer_heights,
    calculate_gpp,
    calculate_number_of_canopy_layers,
    calculate_total_canopy_A_cp,
    calculate_total_community_crown_area,
)
from pyrealm.canopy_model.model.community import Community


class Canopy:
    """A class containing attributes of a canopy, including the structure."""

    def __init__(self, community: Community, canopy_gap_fraction: float) -> None:
        self.max_individual_height = community.t_model_heights.max()

        self.total_community_crown_area = calculate_total_community_crown_area(
            community
        )

        self.number_of_canopy_layers = calculate_number_of_canopy_layers(
            community.cell_area, self.total_community_crown_area, canopy_gap_fraction
        )

        self.canopy_layer_heights = calculate_canopy_layer_heights(
            self.number_of_canopy_layers,
            self.max_individual_height,
            community,
            canopy_gap_fraction,
        )

        # TODO there may be a more efficient solution here that does not use a loop.
        self.A_cp_within_layer = map(
            calculate_total_canopy_A_cp,
            self.canopy_layer_heights,
            np.full(self.number_of_canopy_layers, canopy_gap_fraction),
            np.full(self.number_of_canopy_layers, community),
        )

        self.gpp = calculate_gpp(
            np.zeros(self.number_of_canopy_layers),
            np.zeros(self.number_of_canopy_layers),
        )
