import dotmap
import yaml

# TODO - think about being able to load a user preset
# TODO - maybe define here and not in YAML? Probably faster (can convert to
#        numpy and compile).

with open('data/params.yaml') as param:
    PARAM = dotmap.DotMap(yaml.load(param, Loader=yaml.SafeLoader))
