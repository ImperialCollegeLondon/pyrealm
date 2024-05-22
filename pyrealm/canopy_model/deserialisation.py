import json

from community_sketch import Flora, ImportedCommunity, PlantFunctionalType


class FloraDeserialiser:
    @classmethod
    def load_flora(cls, path: str) -> Flora:
        with open(path) as file:
            pfts_json = json.load(file)
        pfts = PlantFunctionalType.schema().load(pfts_json, many=True)
        return Flora(pfts)


class CommunityDeserialiser:
    def __init__(self, flora: Flora):
        self.flora = flora

    def load_communities(self, path: str) -> list[ImportedCommunity]:
        with open(path) as file:
            communities_json = json.load(file)
            flora = self.flora
        imported_communities = ImportedCommunity.schema().load(
            communities_json, many=True
        )
        return imported_communities
