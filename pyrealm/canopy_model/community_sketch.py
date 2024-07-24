"""A very incomplete sketch of some community and demography functionality."""

from pyrealm.canopy_model.deserialisation.deserialisation import (
    CommunityDeserialiser,
    PlantFunctionalTypeDeserialiser,
)
from pyrealm.canopy_model.model.canopy import Canopy

if __name__ == "__main__":
    pfts = PlantFunctionalTypeDeserialiser.load_plant_functional_types(
        "pyrealm_build_data/community/pfts.json"
    )

    cell_area = 32

    communities = CommunityDeserialiser(pfts, cell_area).load_communities(
        "pyrealm_build_data/community/communities.json"
    )

    community_gap_fraction = 2 / 32

    for community in communities:
        canopy = Canopy(community, community_gap_fraction)
        print("canopy layer heights: \n")
        print(canopy.canopy_layer_heights)
