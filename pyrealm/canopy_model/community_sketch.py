"""A very incomplete sketch of some community and demography functionality."""

from pyrealm.canopy_model.base_objects import Flora
from pyrealm.canopy_model.canopy import Canopy
from pyrealm.canopy_model.community import Community
from pyrealm.canopy_model.deserialisation import (
    CommunityDeserialiser,
    PlantFunctionalTypeDeserialiser,
)

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
        print("canopy layer heights: \n")
        print(canopy.canopy_layer_heights)
