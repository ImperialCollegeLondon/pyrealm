"""Functionality for community modelling."""

from dataclasses import dataclass, field

from dataclasses_json import dataclass_json

from pyrealm.canopy_model.model.cohort import Cohort


@dataclass_json
@dataclass
class Community:
    """Represents a community of plants in a single location.

    Dataclass for storing Community data imported from json.
    """

    cell_id: int
    cohorts: list[Cohort]
    cell_area: float
    seedbank: object = field(init=False)
    fruit: object = field(init=False)

    def __post_init__(self) -> None:
        # Things to add later - the fields for seedbank and fruit can be initialised
        # here.

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
