"""Base objects for import to the canopy module."""

from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
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


# class FloraDataFrame(pd.DataFrame):
#     def __init__(self) -> None:
#         super().__init__()
#         #add some validation here
