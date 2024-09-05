"""Base objects for import to the canopy module."""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import marshmallow_dataclass
import pandas as pd
from marshmallow.exceptions import ValidationError

if sys.version_info[:2] >= (3, 11):
    import tomllib
    from tomllib import TOMLDecodeError
else:
    import tomli as tomllib
    from tomli import TOMLDecodeError


@dataclass
class PlantFunctionalType:
    """The PlantFunctionalType dataclass.

    This dataclass implements the set of traits required to define a plant functional
    type for use in ``pyrealm``. The majority of the traits are those required to
    parameterise the T Model :cite:`Li:2014bc`.  The
    default values are taken from Table 1 of :cite:t:`Li:2014bc`.

    Note that the foliage maintenance respiration fraction is not named in the original
    T Model implementation, but has been included as a modifiable trait in this
    implementation. This implementation adds two further canopy shape parameters (``m``
    and ``n``), which are then used to calculate two derived attributes (``q_m`` and
    ``z_max_ratio``).
    """

    name: str
    """The name of the plant functional type."""
    a_hd: float = 116.0
    """Initial slope of height-diameter relationship (:math:`a`, 116.0, -)"""
    ca_ratio: float = 390.43
    """Initial ratio of crown area to stem cross-sectional area
    (:math:`c`, 390.43, -)"""
    h_max: float = 25.33
    """Maximum tree height (:math:`H_m`, 25.33, m)"""
    rho_s: float = 200.0
    r"""Sapwood density (:math:`\rho_s`, 200.0, kg Cm-3)"""
    lai: float = 1.8
    """Leaf area index within the crown (:math:`L`, 1.8, -)"""
    sla: float = 14.0
    r"""Specific leaf area (:math:`\sigma`, 14.0, m2 kg-1 C)"""
    tau_f: float = 4.0
    r"""Foliage turnover time (:math:`\tau_f`, 4.0, years)"""
    tau_r: float = 1.04
    """Fine-root turnover time (:math:`\tau_r`, 1.04, years)"""
    par_ext: float = 0.5
    """Extinction coefficient of photosynthetically active radiation (PAR) (:math:`k`,
    0.5, -)"""
    yld: float = 0.17
    """Yield_factor (:math:`y`, 0.17, -)"""
    zeta: float = 0.17
    r"""Ratio of fine-root mass to foliage area (:math:`\zeta`, 0.17, kg C m-2)"""
    resp_r: float = 0.913
    """Fine-root specific respiration rate (:math:`r_r`, 0.913, year-1)"""
    resp_s: float = 0.044
    """Sapwood-specific respiration rate (:math:`r_s`, 0.044, year-1)"""
    resp_f: float = 0.1
    """Foliage maintenance respiration fraction (:math:`r_f`,  0.1, -)"""
    m: float = 2
    """Canopy shape parameter (:math:`m`, -)"""
    n: float = 5
    """Canopy shape parameter (:math:`n`, -)"""

    q_m: float = field(init=False)
    """Scaling factor to derive maximum crown radius from crown area."""
    z_max_prop: float = field(init=False)
    """Proportion of stem height at which maximum crown radius is found."""

    def __post_init__(self) -> None:
        """Populate derived attributes.

        This method populates the ``q_m`` and ``z_max_ratio`` attributes from the
        provided values of ``m`` and ``n``.
        """

        # Calculate q_m
        self.q_m = calculate_q_m(m=self.m, n=self.n)
        self.z_max_prop = calculate_z_max_proportion(m=self.m, n=self.n)


PlantFunctionalTypeSchema = marshmallow_dataclass.class_schema(PlantFunctionalType)
"""Marshmallow validation schema class for validating PlantFunctionalType data."""


def calculate_q_m(m: float, n: float) -> float:
    """Calculate a q_m value.

    The value of q_m is a constant canopy scaling parameter derived from the ``m`` and
    ``n`` attributes defined for a plant functional type.

    Args:
        m: Canopy shape parameter
        n: Canopy shape parameter
    """
    return (
        m
        * n
        * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
        * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_z_max_proportion(m: float, n: float) -> float:
    """Calculate the z_m proportion.

    The z_m proportion is the constant proportion of stem height at which the maximum
    crown radius is found for a given plant functional type.

    Args:
        m: Canopy shape parameter
        n: Canopy shape parameter
    """

    return ((n - 1) / (m * n - 1)) ** (1 / n)


class Flora(dict[str, PlantFunctionalType]):
    """Defines the flora used in a ``virtual_ecosystem`` model.

    The flora is the set of plant functional types used within a particular simulation
    and this class provides dictionary-like access to a defined set of
    :class:`~virtual_ecosystem.models.plants.functional_types.PlantFunctionalType`
    instances.

    Instances of this class should not be altered during model fitting, at least until
    the point where plant evolution is included in the modelling process.

    Args:
        pfts: A sequence of ``PlantFunctionalType`` instances, which must not have
            duplicated
            :attr:`~virtual_ecosystem.models.plants.functional_types.PlantFunctionalType.pft_name`
            attributes.
    """

    def __init__(self, pfts: Sequence[PlantFunctionalType]) -> None:
        # Initialise the dict superclass to implement dict like behaviour
        super().__init__()

        # Check the PFT data
        if (not isinstance(pfts, Sequence)) or (
            not all([isinstance(v, PlantFunctionalType) for v in pfts])
        ):
            raise ValueError(
                "The pfts argument must be a sequence of PlantFunctionalType instances"
            )

        # Validate the PFT instances - check there are no duplicate PFT names.
        pft_names = Counter([p.name for p in pfts])
        duplicates = [k for k, v in pft_names.items() if v > 1]

        if duplicates:
            raise ValueError(
                f"Duplicated plant functional type names: {','.join(duplicates)}"
            )

        # Populate the dictionary using the PFT name as key
        for name, pft in zip(pft_names, pfts):
            self[name] = pft

    @classmethod
    def _from_file_data(cls, file_data: dict) -> Flora:
        """Create a Flora object from a JSON string.

        Args:
            file_data: The payload from a data file defining plant functional types.
        """
        try:
            pfts = PlantFunctionalTypeSchema().load(file_data["pft"], many=True)  # type: ignore[attr-defined]
        except ValidationError as excep:
            raise excep

        return cls(pfts=pfts)

    @classmethod
    def from_json(cls, path: Path) -> Flora:
        """Create a Flora object from a JSON file.

        Args:
            path: A path to a JSON file of plant functional type definitions.
        """

        try:
            file_data = json.load(open(path))
        except (FileNotFoundError, json.JSONDecodeError) as excep:
            raise excep

        return cls._from_file_data(file_data=file_data)

    @classmethod
    def from_toml(cls, path: Path) -> Flora:
        """Create a Flora object from a TOML file.

        Args:
            path: A path to a TOML file of plant functional type definitions.
        """

        try:
            file_data = tomllib.load(open(path, "rb"))
        except (FileNotFoundError, TOMLDecodeError) as excep:
            raise excep

        return cls._from_file_data(file_data)

    @classmethod
    def from_csv(cls, path: Path) -> Flora:
        """Create a Flora object from a CSV file.

        Args:
            path: A path to a CSV file of plant functional type definitions.
        """

        try:
            data = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.ParserError) as excep:
            raise excep

        return cls._from_file_data({"pft": data.to_dict(orient="records")})
