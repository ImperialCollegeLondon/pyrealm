"""A very incomplete sketch of some community and demography functionality."""

from pyrealm.canopy_model.deserialisation.deserialisation import (
    CommunityDeserialiser,
    PlantFunctionalTypeDeserialiser,
)
from pyrealm.canopy_model.model.canopy import Canopy

# from pyrealm.canopy_model.model.canopy import Canopy
# from pyrealm.canopy_model.model.community import Community

if __name__ == "__main__":

    pfts = PlantFunctionalTypeDeserialiser.load_plant_functional_types(
        "pyrealm_build_data/community/pfts.json"
    )

    communities = CommunityDeserialiser(pfts, 32).load_communities(
        "pyrealm_build_data/community/communities.json"
    )

    for community in communities:
        canopy = Canopy(community, 2 / 32)
        print("canopy layer heights: \n")
        print(canopy.canopy_layer_heights)
