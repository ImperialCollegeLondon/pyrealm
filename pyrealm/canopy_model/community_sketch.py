"""A very incomplete sketch of some community and demography functionality."""

from pyrealm.canopy_model.deserialisation.deserialisation import (
    PlantFunctionalTypeDeserialiser,
)
from pyrealm.canopy_model.model.canopy import Canopy
from pyrealm.canopy_model.model.community import Community
from pyrealm.canopy_model.model.flora import Flora

if __name__ == "__main__":
    pfts = PlantFunctionalTypeDeserialiser.load_plant_functional_types(
        "pyrealm_build_data/community/pfts.json"
    )

    flora = Flora(pfts)

    # Would like to add these to config somewhere
    cell_area = 32
    community_gap_fraction = 2 / 32
    csv_path = "pyrealm_build_data/community/communities.csv"

    communities = Community.load_communities_from_csv(cell_area, csv_path, flora)

    for community in communities:
        canopy = Canopy(community, community_gap_fraction)
        print("canopy layer heights: \n")
        print(canopy.canopy_layer_heights)
