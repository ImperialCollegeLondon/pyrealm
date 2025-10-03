"""The flora module implements definitions of:

* The ``PlantFunctionalType`` and ``PlantFunctionalTypeStrict`` dataclasses, which are
  used to parameterise the traits of different plant functional types. The
  ``PlantFunctionalType`` dataclass is a subclass of ``PlantFunctionalTypeStrict`` that
  simply adds default values to the attributes.
* The ``PlantFunctionalTypeStrict`` dataclass is used as the basis of a
  :mod:`~marshmallow` schema for validating the creation of plant functional types from
  data files. This intentionally enforces a complete description of the traits in the
  input data. The ``PlantFunctionalType`` is provided as a more convenient API for
  programmatic use.
* The Flora dataclass, which represents a collection of plant functional types for use
  in describing a plant community in a simulation. It provides the same trait attributes
  as the plant functional type classes, but the values are arrays of trait values across
  the provided PFTS. The Flora class also defines factory methods to create instances
  from plant functional type data stored in JSON, TOML or CSV formats.
* The StemTraits dataclass, which represents a collection of stems used in a simulation.
  It again provides the same trait attributes as the plant functional type classes, but
  as arrays. This differs from the Flora class in allowing multiple stems of the same
  plant functional type and is primarily used to broadcast PFT traits into arrays for
  use in calculating demography across the stems in plant cohorts.
"""  # noqa: D415

from __future__ import annotations

import json
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field, fields
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import ClassVar

import marshmallow_dataclass
import numpy as np
import pandas as pd
from marshmallow.exceptions import ValidationError
from numpy.typing import NDArray

from pyrealm.demography.core import (
    CohortMethods,
    PandasExporter,
    _validate_demography_array_arguments,
)


def calculate_crown_q_m(
    m: float | NDArray[np.float64], n: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    """Calculate the crown scaling trait ``q_m``.

    The value of q_m is a constant crown scaling parameter derived from the ``m`` and
    ``n`` attributes defined for a plant functional type.

    Args:
        m: Crown shape parameter
        n: Crown shape parameter
    """
    return (
        m
        * n
        * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
        * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_crown_z_max_proportion(
    m: float | NDArray[np.float64], n: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    r"""Calculate the z_m trait.

    The z_m proportion (:math:`p_{zm}`) is the constant proportion of stem height at
    which the maximum crown radius is found for a given plant functional type.

    .. math::

        p_{zm} = \left(\dfrac{n-1}{m n -1}\right)^ {\tfrac{1}{n}}

    Args:
        m: Crown shape parameter
        n: Crown shape parameter
    """

    return ((n - 1) / (m * n - 1)) ** (1 / n)


@dataclass(frozen=True)
class PlantFunctionalTypeStrict:
    """The PlantFunctionalTypeStrict dataclass.

    This dataclass implements the set of traits required to define a plant functional
    type for use in ``pyrealm``.

    * Most traits are taken from the definition of the T Model of plant growth and GPP
      allocation :cite:`Li:2014bc`.
    * The foliage maintenance respiration fraction was not explicitly included in
      :cite:t:`Li:2014bc` - there was assumed to be a 10% penalty on GPP before
      calculating the other component - but has been explicitly included here.
    * This implementation adds two further crown shape parameters (``m`` and ``n`` and
      ``f_g``). The first two are then used to calculate two constant derived attributes
      (``q_m`` and ``z_max_ratio``) that define the vertical distribution of the crown.
      The last parameter (``f_g``) is the crown gap fraction, that defines the vertical
      distribution of leaves within the crown. This crown model parameterisation
      follows the implementation developed in the PlantFATE model :cite:`joshi:2022a`.

    See also :class:`~pyrealm.demography.flora.PlantFunctionalType` for the default
    values implemented in that subclass.
    """

    name: str
    r"""The name of the plant functional type."""
    a_hd: float
    r"""Initial slope of height-diameter relationship (:math:`a`, -)"""
    ca_ratio: float
    r"""Initial ratio of crown area to stem cross-sectional area
    (:math:`c`, -)"""
    h_max: float
    r"""Maximum tree height (:math:`H_m`, m)"""
    rho_s: float
    r"""Sapwood density (:math:`\rho_s`, kg Cm-3)"""
    lai: float
    """Leaf area index within the crown (:math:`L`,  -)"""
    sla: float
    r"""Specific leaf area (:math:`\sigma`,  m2 kg-1 C)"""
    tau_f: float
    r"""Foliage turnover time (:math:`\tau_f`,years)"""
    tau_rt: float
    r"""Reproductive tissue turnover time (:math:`\tau_rt`,years)"""
    tau_r: float
    r"""Fine-root turnover time (:math:`\tau_r`,  years)"""
    par_ext: float
    r"""Extinction coefficient of photosynthetically active radiation (PAR) (:math:`k`,
     -)"""
    yld: float
    r"""Yield factor (:math:`y`,  -)"""
    zeta: float
    r"""Ratio of fine-root mass to foliage area (:math:`\zeta`, kg C m-2)"""
    resp_r: float
    r"""Fine-root specific respiration rate (:math:`r_r`, year-1)"""
    resp_rt: float
    r"""Reproductive tissue respiration rate (:math:`r_{rt}`, year-1)"""
    resp_s: float
    r"""Sapwood-specific respiration rate (:math:`r_s`,  year-1)"""
    resp_f: float
    r"""Foliage maintenance respiration fraction (:math:`r_f`,  -)"""
    m: float
    r"""Crown shape parameter (:math:`m`, -)"""
    n: float
    r"""Crown shape parameter (:math:`n`, -)"""
    f_g: float
    r"""Crown gap fraction (:math:`f_g`, -)"""

    q_m: float = field(init=False)
    """Scaling factor to derive maximum crown radius from crown area."""
    z_max_prop: float = field(init=False)
    """Proportion of stem height at which maximum crown radius is found."""

    p_foliage_for_reproductive_tissue: float
    """Mass of reproductive tissue as a proportion of foliage mass (:math:`p_{rt}`, -).
    """
    gpp_topslice: float
    """Proportion of GPP to topslice before allocation."""

    def __post_init__(self) -> None:
        """Populate derived attributes.

        This method populates the ``q_m`` and ``z_max_ratio`` attributes from the
        provided values of ``m`` and ``n``.
        """

        # Calculate q_m and z_max proportion. Need to use __setattr__ because the
        # dataclass is frozen.
        object.__setattr__(self, "q_m", calculate_crown_q_m(m=self.m, n=self.n))
        object.__setattr__(
            self, "z_max_prop", calculate_crown_z_max_proportion(m=self.m, n=self.n)
        )


@dataclass(frozen=True)
class PlantFunctionalType(PlantFunctionalTypeStrict):
    r"""The PlantFunctionalType dataclass.

    This dataclass is a subclass of
    :class:`~pyrealm.demography.flora.PlantFunctionalTypeStrict` that implements exactly
    the same set of traits but provides default values. This class is intended as a
    convenience API for programmatic use, where the parent provides a strict schema for
    generating plant functional type instances from data.

    The table below lists the attributes and default values taken from Table 1 of
    :cite:t:`Li:2014bc`, except for ``m``, ``n`` and ``f_g`` which take representative
    values from :cite:t:`joshi:2022a`.

    .. csv-table::
        :header: "Attribute", "Default", "Unit"
        :widths: 15, 10, 30

        a_hd, 116.0, -
        ca_ratio, 390.43, -
        h_max,  25.33, m
        rho_s, 200.0, kg Cm-3
        lai,  1.8, -
        sla,  14.0, m2 kg-1 C
        tau_f, 4.0, years
        tau_r,  1.04, years
        par_ext, 0.5, -
        yld, 0.6, -
        zeta, 0.17, kg C m-2
        resp_r,  0.913, year-1
        resp_s,  0.044, year-1
        resp_f,  0.1, -
        m, 2, -
        n, 5, -
        f_g, 0.05, -
    """

    name: str
    a_hd: float = 116.0
    ca_ratio: float = 390.43
    h_max: float = 25.33
    rho_s: float = 200.0
    lai: float = 1.8
    sla: float = 14.0
    tau_f: float = 4.0
    tau_rt: float = 1.0  # Default value 1 as this is a denominator
    tau_r: float = 1.04
    par_ext: float = 0.5
    yld: float = 0.6
    zeta: float = 0.17
    resp_r: float = 0.913
    resp_s: float = 0.044
    resp_f: float = 0.1
    resp_rt: float = 0.0
    m: float = 2
    n: float = 5
    f_g: float = 0.05
    p_foliage_for_reproductive_tissue: float = 0.0
    gpp_topslice: float = 0.0


PlantFunctionalTypeSchema = marshmallow_dataclass.class_schema(
    PlantFunctionalTypeStrict
)
"""Marshmallow validation schema class for validating PlantFunctionalType data.

This schema explicitly uses the strict version of the dataclass, which enforces complete
descriptions of plant functional type data rather than allowing partial data and filling
in gaps from the default values.
"""


@dataclass(frozen=True)
class Flora(PandasExporter):
    """A dataclass providing trait data on collection of plant functional types.

    A flora provides trait data on the complete collection of plant functional types
    that will be used within a particular simulation. The dataclass provides access
    to trait attributes as row arrays across those plant functional types.

    The class is created using a list of
    :class:`~pyrealm.demography.flora.PlantFunctionalType` or
    :class:`~pyrealm.demography.flora.PlantFunctionalTypeStrict` instances, which must
    have unique names.

    Args:
        pfts: A sequence of ``PlantFunctionalType`` or ``PlantFunctionalTypeStrict``
            instances, which must not have duplicated
            :attr:`~pyrealm.demography.flora.PlantFunctionalTypeStrict.name` attributes.
    """

    # The only init argument.
    pfts: InitVar[Sequence[PlantFunctionalTypeStrict]]
    r"""A sequence of plant functional type instances to include in the Flora."""

    # A class variable setting the names of PFT traits held as arrays.
    array_attrs: ClassVar[tuple[str, ...]] = tuple(
        f.name for f in fields(PlantFunctionalTypeStrict)
    )

    # Populated post init
    # - trait arrays
    name: NDArray[np.str_] = field(init=False)
    r"""The name of the plant functional type."""
    a_hd: NDArray[np.float64] = field(init=False)
    r"""Initial slope of height-diameter relationship (:math:`a`, -)"""
    ca_ratio: NDArray[np.float64] = field(init=False)
    r"""Initial ratio of crown area to stem cross-sectional area
    (:math:`c`, -)"""
    h_max: NDArray[np.float64] = field(init=False)
    r"""Maximum tree height (:math:`H_m`, m)"""
    rho_s: NDArray[np.float64] = field(init=False)
    r"""Sapwood density (:math:`\rho_s`, kg Cm-3)"""
    lai: NDArray[np.float64] = field(init=False)
    """Leaf area index within the crown (:math:`L`,  -)"""
    sla: NDArray[np.float64] = field(init=False)
    r"""Specific leaf area (:math:`\sigma`,  m2 kg-1 C)"""
    tau_f: NDArray[np.float64] = field(init=False)
    r"""Foliage turnover time (:math:`\tau_f`,years)"""
    tau_r: NDArray[np.float64] = field(init=False)
    r"""Fine-root turnover time (:math:`\tau_r`,  years)"""
    tau_rt: NDArray[np.float64] = field(init=False)
    r"""Reproductive tissue turnover time (:math:`\tau_rt`,years)"""
    par_ext: NDArray[np.float64] = field(init=False)
    r"""Extinction coefficient of photosynthetically active radiation (PAR) (:math:`k`,
     -)"""
    yld: NDArray[np.float64] = field(init=False)
    r"""Yield factor (:math:`y`,  -)"""
    zeta: NDArray[np.float64] = field(init=False)
    r"""Ratio of fine-root mass to foliage area (:math:`\zeta`, kg C m-2)"""
    resp_r: NDArray[np.float64] = field(init=False)
    r"""Fine-root specific respiration rate (:math:`r_r`, year-1)"""
    resp_s: NDArray[np.float64] = field(init=False)
    r"""Sapwood-specific respiration rate (:math:`r_s`,  year-1)"""
    resp_f: NDArray[np.float64] = field(init=False)
    r"""Foliage maintenance respiration fraction (:math:`r_f`,  -)"""
    resp_rt: NDArray[np.float64] = field(init=False)
    r"""Reproductive tissue respiration rate (:math:`r_{rt}`,  -)"""
    m: NDArray[np.float64] = field(init=False)
    r"""Crown shape parameter (:math:`m`, -)"""
    n: NDArray[np.float64] = field(init=False)
    r"""Crown shape parameter (:math:`n`, -)"""
    f_g: NDArray[np.float64] = field(init=False)
    r"""Crown gap fraction (:math:`f_g`, -)"""
    q_m: NDArray[np.float64] = field(init=False)
    """Scaling factor to derive maximum crown radius from crown area."""
    z_max_prop: NDArray[np.float64] = field(init=False)
    """Proportion of stem height at which maximum crown radius is found."""

    # - other instance attributes
    pft_dict: dict[str, PlantFunctionalTypeStrict] = field(init=False)
    """A dictionary of the original plant functional type instances, keyed by name."""
    pft_indices: dict[str, int] = field(init=False)
    """An dictionary giving the index of each PFT name in the trait array attributes."""
    n_pfts: int = field(init=False)
    """The number of plant functional types in the Flora instance."""
    _n_stems: int = field(init=False)
    """Private attribute for compatibility with StemTraits API."""

    p_foliage_for_reproductive_tissue: NDArray[np.float64] = field(init=False)
    """Proportion of foliage used to calculate reproductive tissue."""
    gpp_topslice: NDArray[np.float64] = field(init=False)
    """Proportion of GPP to topslice before allocation."""

    def __post_init__(self, pfts: Sequence[PlantFunctionalTypeStrict]) -> None:
        # Check the PFT data
        if (not isinstance(pfts, Sequence)) or (
            not all([isinstance(v, PlantFunctionalTypeStrict) for v in pfts])
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

        # Populate the pft dictionary using the PFT name as key and the number of PFTs
        object.__setattr__(self, "pft_dict", {p.name: p for p in pfts})
        object.__setattr__(self, "n_pfts", len(pfts))
        object.__setattr__(self, "_n_stems", len(pfts))

        # Populate the trait attributes with arrays
        for pft_field in self.array_attrs:
            object.__setattr__(
                self, pft_field, np.array([getattr(pft, pft_field) for pft in pfts])
            )

        # Populate the pft trait indices
        object.__setattr__(self, "pft_indices", {v: k for k, v in enumerate(self.name)})

    def __repr__(self) -> str:
        """Simple representation of the Flora instance."""

        return f"Flora with {self._n_stems} functional types: {', '.join(self.name)}"

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

    def get_stem_traits(self, pft_names: NDArray[np.str_]) -> StemTraits:
        """Generates a stem traits object for a set of names.

        Args:
            pft_names: An array of PFT names for each stem.
        """

        # Check the initial PFT values are known
        unknown_pfts = set(pft_names).difference(self.name)

        if unknown_pfts:
            raise ValueError(
                f"Plant functional types unknown in flora: {','.join(unknown_pfts)}"
            )

        # Get the indices for the cohort PFT names in the flora PFT data
        pft_index = [self.pft_indices[str(nm)] for nm in pft_names]

        # For each trait, get that attribute from the Flora, extract the values matching
        # the pft_names and pass that into the StemTraits constructor. Validation is
        # turned off here, because the creation method guarantees the data is properly
        # formatted.
        return StemTraits(
            **{trt: getattr(self, trt)[pft_index] for trt in self.array_attrs},
            validate=False,
        )


@dataclass()
class StemTraits(PandasExporter, CohortMethods):
    """A dataclass for stem traits.

    This dataclass is used to provide arrays of plant functional type (PFT) traits
    across a set of stems. The main use case is to provide stem trait data as arrays
    across the cohorts within a community object.

    It provides the same attribute interface as the
    :class:`~pyrealm.demography.flora.Flora` class, but unlike that class:

    * is purely a data container, and
    * plant functional types can be represented multiple times to represent multiple
      stems or cohorts of the same PFT.
    """

    # A class variable setting the attribute names of traits.
    array_attrs: ClassVar[tuple[str, ...]] = tuple(
        f.name for f in fields(PlantFunctionalTypeStrict)
    )
    count_attr: ClassVar[str] = "_n_stems"

    # Instance trait attributes
    name: NDArray[np.str_]
    r"""The name of the plant functional type."""
    a_hd: NDArray[np.float64]
    r"""Initial slope of height-diameter relationship (:math:`a`, -)"""
    ca_ratio: NDArray[np.float64]
    r"""Initial ratio of crown area to stem cross-sectional area
    (:math:`c`, -)"""
    h_max: NDArray[np.float64]
    r"""Maximum tree height (:math:`H_m`, m)"""
    rho_s: NDArray[np.float64]
    r"""Sapwood density (:math:`\rho_s`, kg Cm-3)"""
    lai: NDArray[np.float64]
    """Leaf area index within the crown (:math:`L`,  -)"""
    sla: NDArray[np.float64]
    r"""Specific leaf area (:math:`\sigma`,  m2 kg-1 C)"""
    tau_f: NDArray[np.float64]
    r"""Foliage turnover time (:math:`\tau_f`,years)"""
    tau_rt: NDArray[np.float64]
    r"""Reproductive tissue turnover time (:math:`\tau_rt`,years)"""
    tau_r: NDArray[np.float64]
    r"""Fine-root turnover time (:math:`\tau_r`,  years)"""
    par_ext: NDArray[np.float64]
    r"""Extinction coefficient of photosynthetically active radiation (PAR) (:math:`k`,
     -)"""
    yld: NDArray[np.float64]
    r"""Yield factor (:math:`y`,  -)"""
    zeta: NDArray[np.float64]
    r"""Ratio of fine-root mass to foliage area (:math:`\zeta`, kg C m-2)"""
    resp_r: NDArray[np.float64]
    r"""Fine-root specific respiration rate (:math:`r_r`, year-1)"""
    resp_s: NDArray[np.float64]
    r"""Sapwood-specific respiration rate (:math:`r_s`,  year-1)"""
    resp_f: NDArray[np.float64]
    r"""Foliage maintenance respiration fraction (:math:`r_f`,  -)"""
    resp_rt: NDArray[np.float64]
    r"""Reproductive tissue respiration rate (:math:`r_{rt}`,  -)"""
    m: NDArray[np.float64]
    r"""Crown shape parameter (:math:`m`, -)"""
    n: NDArray[np.float64]
    r"""Crown shape parameter (:math:`n`, -)"""
    f_g: NDArray[np.float64]
    r"""Crown gap fraction (:math:`f_g`, -)"""
    q_m: NDArray[np.float64]
    """Scaling factor to derive maximum crown radius from crown area."""
    z_max_prop: NDArray[np.float64]
    """Proportion of stem height at which maximum crown radius is found."""

    p_foliage_for_reproductive_tissue: NDArray[np.float64]
    """Proportion of foliage used to calculate reproductive tissue."""
    gpp_topslice: NDArray[np.float64]
    """Proportion of GPP to topslice before allocation."""

    validate: bool = True
    """Boolean flag to control validation of the input array sizes."""

    # Post init attributes
    _n_stems: int = field(init=False)

    def __post_init__(self) -> None:
        """Post init validation and attribute setting."""

        if self.validate:
            _validate_demography_array_arguments(
                trait_args={k: getattr(self, k) for k in self.array_attrs}
            )

        self._n_stems = len(self.a_hd)
