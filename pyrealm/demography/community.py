"""This modules provides the Community class, which contains the set of size-structured
cohorts of plants across a range of plant functional types that occur a given location
(or 'cell') with a given cell id number and area.

The class provides factory methods to create Community instances from CSV, JSON and TOML
files, using :mod:`marshmallow` schemas to both validate the input data and to perform
post processing to align the input formats to the initialisation arguments to the
Community class.

Internally, the cohort data in the Community class is represented as a dictionary of
`numpy` arrays.

Worked example
==============

The example code below demonstrates defining PFTs, creating a Flora collection,
initializing a Community, and computing ecological metrics using the T Model for a set
of plant cohorts.

>>> import pandas as pd
>>>
>>> from pyrealm.demography.flora import PlantFunctionalType, Flora
>>> from pyrealm.demography.tmodel import (
...     calculate_heights, calculate_crown_areas, calculate_stem_masses,
...     calculate_foliage_masses
... )

>>> pft1 = PlantFunctionalType(
...     name="Evergreen Tree",
...     a_hd=120.0,
...     ca_ratio=380.0,
...     h_max=30.0,
...     rho_s=210.0,
...     lai=3.0,
...     sla=12.0,
...     tau_f=5.0,
...     tau_r=1.2,
...     par_ext=0.6,
...     yld=0.65,
...     zeta=0.18,
...     resp_r=0.95,
...     resp_s=0.045,
...     resp_f=0.12,
...     m=2.5,
...     n=4.5,
... )

>>> pft2 = PlantFunctionalType(
...     name="Deciduous Shrub",
...     a_hd=100.0,
...     ca_ratio=350.0,
...     h_max=4.0,
...     rho_s=180.0,
...     lai=2.0,
...     sla=15.0,
...     tau_f=3.0,
...     tau_r=0.8,
...     par_ext=0.4,
...     yld=0.55,
...     zeta=0.15,
...     resp_r=0.85,
...     resp_s=0.05,
...     resp_f=0.1,
...     m=3.0,
...     n=5.0,
... )

Create a Flora collection:

>>> flora = Flora([pft1, pft2])

Define community data as size-structured cohorts of given plant functional types with a
given number of individuals.

>>> cohort_dbh_values = np.array([0.10, 0.03, 0.12, 0.025])
>>> cohort_n_individuals = np.array([100, 200, 150, 180])
>>> cohort_pft_names = np.array(
...    ["Evergreen Tree", "Deciduous Shrub", "Evergreen Tree", "Deciduous Shrub"]
... )

Initialize a Community into an area of 1000 square meter with the given cohort data:

>>> community = Community(
...     cell_id=1,
...     cell_area=1000.0,
...     flora=flora,
...     cohorts=Cohorts(
...         dbh_values=cohort_dbh_values,
...         n_individuals=cohort_n_individuals,
...         pft_names=cohort_pft_names,
...     ),
... )

The data in the Community class is stored under three attributes, each of which stores
an instance of a dataclass holding related parts of the community data. All have a
``to_pandas`` method that can be used to visualise and explore the data. Note that the
`Cohorts` class automatically adds a unique internal ID to each cohort which is not
shown here:

>>> community.cohorts.to_pandas().drop(columns="cohort_id")
   dbh_values  n_individuals        pft_names
0       0.100            100   Evergreen Tree
1       0.030            200  Deciduous Shrub
2       0.120            150   Evergreen Tree
3       0.025            180  Deciduous Shrub

>>> community.stem_allometry.to_pandas()[
...     ["stem_height", "crown_area", "stem_mass", "crown_r0", "crown_z_max"]
... ]  # doctest: +NORMALIZE_WHITESPACE
                   stem_height  crown_area  stem_mass  crown_r0  crown_z_max
column_stem_index
0                     9.890399    2.459835   8.156296  0.339477     7.789552
1                     2.110534    0.174049   0.134266  0.083788     1.642777
2                    11.436498    3.413238  13.581094  0.399890     9.007241
3                     1.858954    0.127752   0.082126  0.071784     1.446955

>>> community.stem_traits.to_pandas()[
...     ["name", "a_hd", "ca_ratio", "sla", "par_ext", "q_m",  "z_max_prop"]
... ]
              name   a_hd  ca_ratio   sla  par_ext       q_m  z_max_prop
0   Evergreen Tree  120.0     380.0  12.0      0.6  2.606561    0.787587
1  Deciduous Shrub  100.0     350.0  15.0      0.4  2.809188    0.778371
2   Evergreen Tree  120.0     380.0  12.0      0.6  2.606561    0.787587
3  Deciduous Shrub  100.0     350.0  15.0      0.4  2.809188    0.778371
"""  # noqa: D205

from __future__ import annotations

import json
import tomllib
import uuid
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from marshmallow import Schema, fields, post_load, validate, validates_schema
from marshmallow.exceptions import ValidationError
from numpy.typing import NDArray

from pyrealm.core.experimental import warn_experimental
from pyrealm.core.utilities import check_input_shapes
from pyrealm.demography.core import CohortMethods, PandasExporter
from pyrealm.demography.flora import Flora, StemTraits
from pyrealm.demography.tmodel import StemAllometry


@dataclass
class Cohorts(PandasExporter, CohortMethods):
    """A dataclass to hold data for a set of plant cohorts.

    The attributes should be numpy arrays of equal length, containing an entry for each
    cohort in the data class. The class automatically populates each cohort with a
    unique id in the `cohort_id` property, using UUID4 values. The setter for this
    property enforces unique values in this property. The most likely source of this
    error would be if the
    :meth:`CohortMethods.add_cohort_data<pyrealm.demography.core.CohortMethods.add_cohort_data>`
    method was used to add a Cohorts instance to itself.
    """

    # A class variable setting the attribute names of traits.
    array_attrs: ClassVar[tuple[str, ...]] = tuple(
        ["dbh_values", "n_individuals", "pft_names", "cohort_id"]
    )
    count_attr: ClassVar[str] = "n_cohorts"

    # Instance attributes
    n_individuals: NDArray[np.int_]
    pft_names: NDArray[np.str_]
    _cohort_id: NDArray[np.str_] = field(init=False)
    _dbh_values: NDArray[np.float64] = field(init=False)
    n_cohorts: int = field(init=False)
    dbh_values: InitVar[NDArray[np.float64]]

    __experimental__ = True

    def __post_init__(self, dbh_values: NDArray[np.float64]) -> None:
        """Validation of cohorts data."""

        # TODO - validation - maybe make this optional to reduce workload within
        # simulations

        warn_experimental("Cohorts")

        # Check cohort data types
        if not (
            isinstance(dbh_values, np.ndarray)
            and isinstance(self.n_individuals, np.ndarray)
            and isinstance(self.pft_names, np.ndarray)
        ):
            raise ValueError("Cohort data not passed as numpy arrays")

        # Check the cohort inputs are of equal length
        try:
            check_input_shapes(self.pft_names, self.n_individuals, dbh_values)
        except ValueError:
            raise ValueError("Cohort arrays are of unequal length")

        # Set the DBH values to trigger validation
        setattr(self, "dbh_values", dbh_values)

        # Additional attributes
        self.n_cohorts = dbh_values.size
        self._cohort_id = np.array([str(uuid.uuid4()) for _ in range(self.n_cohorts)])

    @property
    def cohort_id(self) -> NDArray[np.str_]:
        """Automatically populated with a UUID4 id for each cohort."""
        return self._cohort_id

    @cohort_id.setter
    def cohort_id(self, values: NDArray[np.str_]) -> None:
        if len(set(values)) < len(values):
            raise ValueError("Cohort object with duplicated cohort ids")

        self._cohort_id = values

    # Ignoring this redefinition - the name is used above as an init-only argument that
    # is then redefined as a class property with a setter protecting from negative
    # values.

    @property  # type: ignore [no-redef]
    def dbh_values(self) -> NDArray[np.float64]:
        """The diameter at breast height of the cohorts (m)."""
        return self._dbh_values

    @dbh_values.setter
    def dbh_values(self, values: NDArray[np.float64]) -> None:
        """Setter function for DBH values, enforcing strictly positive values."""
        if np.any(values <= 0):
            raise ValueError("DBH values must be strictly positive")

        self._dbh_values = values


class CohortSchema(Schema):
    """A validation schema for Cohort data objects.

    This schema can be used to validate the ``cohorts`` components of JSON and TOML
    community data files, which are simple dictionaries:

    .. code-block:: toml
        :caption: TOML

        dbh_value = 0.2
        n_individuals = 6
        pft_name = "broadleaf"

    .. code-block:: json
        :caption: JSON

        {
            "pft_name": "broadleaf",
            "dbh_value": 0.2,
            "n_individuals": 6
        }
    """

    dbh_value = fields.Float(
        required=True, validate=validate.Range(min=0, min_inclusive=False)
    )
    n_individuals = fields.Integer(
        strict=True, required=True, validate=validate.Range(min=0, min_inclusive=False)
    )
    pft_name = fields.Str(required=True)


class CommunityStructuredDataSchema(Schema):
    """A validation schema for Cohort data in a structured format (JSON/TOML).

    This schema can be used to validate data for creating a Community instance stored in
    a structured format such as JSON or TOML. The format is expected to provide a cell
    area and id along with an array of cohort objects providing the plant functional
    type name, diameter at breast height (DBH) and number of individuals (see
    :class:`~pyrealm.demography.community.CohortSchema`). Example inputs with this
    structure are:

    .. code-block:: toml
        :caption: TOML

        cell_area = 100
        cell_id = 1

        [[cohorts]]
        dbh_value = 0.2
        n_individuals = 6
        pft_name = "broadleaf"

        [[cohorts]]
        dbh_value = 0.25
        n_individuals = 6
        pft_name = "conifer"

    .. code-block:: json
        :caption: JSON

        {
        "cell_id": 1,
        "cell_area": 100,
        "cohorts": [
            {
                "pft_name": "broadleaf",
                "dbh_value": 0.2,
                "n_individuals": 6
            },
            {
                "pft_name": "broadleaf",
                "dbh_value": 0.25,
                "n_individuals": 6
            }]
        }

    Any data validated with this schema is post-processed to convert the cohort objects
    into the arrays of cohort data required to initialise instances of the
    :class:`~pyrealm.demography.community.Community` class.
    """

    cell_id = fields.Integer(required=True, strict=True, validate=validate.Range(min=0))
    cell_area = fields.Float(
        required=True, validate=validate.Range(min=0, min_inclusive=False)
    )
    cohorts = fields.List(
        fields.Nested(CohortSchema), required=True, validate=validate.Length(min=1)
    )

    @post_load
    def convert_to_community_args(self, data: dict, **kwargs: Any) -> dict[str, Any]:
        """Convert cohorts to arrays.

        This post load method converts the cohort arrays into a Cohorts objects and
        packages the data up into the required arguments used to initialise a Community
        object.

        Args:
            data: Data passed to the validator
            kwargs: Additional keyword arguments passed by marshmallow
        """

        return dict(
            cell_id=data["cell_id"],
            cell_area=data["cell_area"],
            cohorts=Cohorts(
                dbh_values=np.array([c["dbh_value"] for c in data["cohorts"]]),
                n_individuals=np.array([c["n_individuals"] for c in data["cohorts"]]),
                pft_names=np.array([c["pft_name"] for c in data["cohorts"]]),
            ),
        )


class CommunityCSVDataSchema(Schema):
    """A validation schema for community initialisation data in CSV format.

    This schema can be used to validate data for creating a Community instance stored in
    CSV format. The file is expected to provide fields providing cell id and cell area
    and then functional type name, diameter at breast height (DBH) and number of
    individuals. Each row is taken to represent a cohort and the cell id and area
    *must** be consistent across rows.

    .. code-block::

        cell_id,cell_area,cohort_pft_names,cohort_dbh_values,cohort_n_individuals
        1,100,broadleaf,0.2,6
        1,100,broadleaf,0.25,6
        1,100,broadleaf,0.3,3
        1,100,broadleaf,0.35,1
        1,100,conifer,0.5,1
        1,100,conifer,0.6,1

    The input data is expected to be provided to this schema as a dictionary of lists of
    field values keyed by field name, as for example by using
    :meth:`pandas.DataFrame.to_dict` with the ``orient='list'`` argument.

    The schema automatically validates that the cell id and area are consistent and then
    post-processing is used to simplify those fields to the scalar inputs required to
    initialise instances of the  :class:`~pyrealm.demography.community.Community` class
    and to convert the cohort data into arrays,
    """

    cell_id = fields.List(fields.Integer(strict=True), required=True)
    cell_area = fields.List(fields.Float(), required=True)
    cohort_dbh_values = fields.List(fields.Float(), required=True)
    cohort_n_individuals = fields.List(fields.Integer(strict=True), required=True)
    cohort_pft_names = fields.List(fields.Str(), required=True)

    @validates_schema
    def validate_consistent_cell_data(self, data: dict, **kwargs: Any) -> None:
        """Schema wide validation.

        Args:
            data: Data passed to the validator
            kwargs: Additional keyword arguments passed by marshmallow
        """

        # Check cell area and cell id consistent
        if not all([c == data["cell_id"][0] for c in data["cell_id"]]):
            raise ValueError(
                "Multiple cell id values fields in community data, see load_communities"
            )

        if not all([c == data["cell_area"][0] for c in data["cell_area"]]):
            raise ValueError("Cell area varies in community data")

    @post_load
    def convert_to_community_args(self, data: dict, **kwargs: Any) -> dict[str, Any]:
        """Make cell data scalar.

        This post load method reduces the repeated cell id and cell area across CSV data
        rows into the scalar inputs required to initialise a Community object and
        packages the data on individual cohorts into a Cohorts object.
        """

        return dict(
            cell_id=data["cell_id"][0],
            cell_area=data["cell_area"][0],
            cohorts=Cohorts(
                dbh_values=np.array(data["cohort_dbh_values"]),
                n_individuals=np.array(data["cohort_n_individuals"]),
                pft_names=np.array(data["cohort_pft_names"]),
            ),
        )


@dataclass
class Community:
    """The plant community class.

    A community is a set of size-structured plant cohorts in a given location, where the
    location has a specified numeric id and a known area in square meters.

    A cohort defines a number of individual plants with the same diameter at breast
    height (DBH) and plant functional type (PFT). Internally, the cohort data is built
    into a :class:`pandas.DataFrame` with each row representing a cohort and each column
    representing a property of the cohort. The initial input data is extended to include
    the plant functional type traits for each cohort (see
    :class:`~pyrealm.demography.flora.PlantFunctionalType`) and then is further extended
    to include the geometric and canopy predictions of the T Model for each cohort.

    Factory methods are provided to load community data from csv, TOML or JSON files.

    Args:
        cell_id: An positive integer id for the community location.
        cell_area: An area in square metres for the community location.
        flora: A flora object containing the plant functional types for the community
        cohort_dbh_values: A numpy array giving the diameter at breast height in metres
            for each cohort.
        cohort_n_individuals: A numpy array giving the number of individuals in each
            cohort.
        cohort_pft_names: A numpy array giving the name of the plant functional type in
            each cohort.
    """

    # Dataclass attributes for initialisation
    # - community wide properties
    cell_id: int
    cell_area: float
    flora: Flora

    # - arrays representing properties of cohorts
    cohorts: Cohorts

    # Post init properties
    n_cohorts: int = field(init=False)
    stem_traits: StemTraits = field(init=False)
    stem_allometry: StemAllometry = field(init=False)

    __experimental__ = True

    def __post_init__(
        self,
    ) -> None:
        """Validate inputs and populate derived community attributes.

        The ``__post_init__`` builds a pandas dataframe of PFT values and T model
        predictions across the validated initial cohort data.
        """

        warn_experimental("Community")

        # Check cell area and cell id
        if not (isinstance(self.cell_area, float | int) and self.cell_area > 0):
            raise ValueError("Community cell area must be a positive number.")

        if not (isinstance(self.cell_id, int) and self.cell_id >= 0):
            raise ValueError("Community cell id must be a integer >= 0.")

        # How many cohorts
        self.n_cohorts = self.cohorts.n_cohorts

        # Get the stem traits for the cohorts
        self.stem_traits = self.flora.get_stem_traits(self.cohorts.pft_names)

        # Populate the stem allometry
        self.stem_allometry = StemAllometry(
            stem_traits=self.stem_traits, at_dbh=self.cohorts._dbh_values
        )

    @classmethod
    def from_csv(cls, path: Path, flora: Flora) -> Community:
        """Create a Community object from a CSV file.

        This factory method checks that the required fields are present in the CSV data
        and that the cell_id and cell_area values are constant. It then passes the data
        through further validation using the
        `meth`:`~pyrealm.demography.community.Community._from_file_data` method and
        returns a Community instance.

        Args:
            path: A path to a CSV file of community data
            flora: A Flora instance providing plant functional types used in the
                community data
        """

        # Load the data
        try:
            file_data = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.ParserError) as excep:
            raise excep

        # Validate the data - there is an inconsequential typing issue here:
        # Argument "data" to "load" of "Schema" has incompatible type
        # "dict[Hashable, Any]";
        # expected "Mapping[str, Any] | Iterable[Mapping[str, Any]]"
        try:
            file_data = CommunityCSVDataSchema().load(data=file_data.to_dict("list"))  # type: ignore[arg-type]
        except ValidationError as excep:
            raise excep

        return cls(**file_data, flora=flora)

    @classmethod
    def from_json(cls, path: Path, flora: Flora) -> Community:
        """Create a Community object from a JSON file.

        This factory method loads community data from a JSON community file and
        validates it using
        `class`:`~pyrealm.demography.community.CommunityStructuredDataSchema` before
        using the data to initialise a Community instance.

        Args:
            path: A path to a JSON file of community data
            flora: A Flora instance providing plant functional types used in the
                community data
        """

        # Load the data
        try:
            file_data = json.load(open(path))
        except (FileNotFoundError, json.JSONDecodeError) as excep:
            raise excep

        # Validate the data
        try:
            file_data = CommunityStructuredDataSchema().load(data=file_data)
        except ValidationError as excep:
            raise excep

        return cls(**file_data, flora=flora)

    @classmethod
    def from_toml(cls, path: Path, flora: Flora) -> Community:
        """Create a Community object from a TOML file.

        This factory method loads community data from a TOML community file and
        validates it using
        `class`:`~pyrealm.demography.community.CommunityStructuredDataSchema` before
        using the data to initialise a Community instance.

        Args:
            path: A path to a TOML file of community data
            flora: A Flora instance providing plant functional types used in the
                community data
        """

        # Load the data
        try:
            file_data = tomllib.load(open(path, "rb"))
        except (FileNotFoundError, TOMLDecodeError) as excep:
            raise excep

        # Validate the data
        try:
            file_data = CommunityStructuredDataSchema().load(data=file_data)
        except ValidationError as excep:
            raise excep

        return cls(**file_data, flora=flora)

    def drop_cohorts(self, drop_indices: NDArray[np.int_]) -> None:
        """Drop cohorts from the community.

        This method drops the identified cohorts from the ``cohorts`` attribute and then
        removes their data from  the ``stem_traits`` and ``stem_allometry`` attributes
        to match.
        """

        self.cohorts.drop_cohort_data(drop_indices=drop_indices)
        self.stem_traits.drop_cohort_data(drop_indices=drop_indices)
        self.stem_allometry.drop_cohort_data(drop_indices=drop_indices)

        self.n_cohorts = self.cohorts.n_cohorts

    def add_cohorts(self, new_data: Cohorts) -> None:
        """Add a new set of cohorts to the community.

        This method extends the ``cohorts`` attribute with the new cohort data and then
        also extends the ``stem_traits`` and ``stem_allometry`` to match.

        Args:
            new_data: An instance of :class:`~pyrealm.demography.community.Cohorts`
                containing cohort data to add to the community.
        """

        self.cohorts.add_cohort_data(new_data=new_data)

        new_stem_traits = self.flora.get_stem_traits(pft_names=new_data.pft_names)
        self.stem_traits.add_cohort_data(new_data=new_stem_traits)

        new_stem_allometry = StemAllometry(
            stem_traits=new_stem_traits, at_dbh=new_data._dbh_values
        )
        self.stem_allometry.add_cohort_data(new_data=new_stem_allometry)

        self.n_cohorts = self.cohorts.n_cohorts

    # @classmethod
    # def load_communities_from_csv(
    #     cls, cell_area: float, csv_path: str, flora: Flora
    # ) -> list[Community]:
    #     """Loads a list of communities from a csv provided in the appropriate format.

    #     The csv should contain the following columns: cell_id,
    #     diameter_at_breast_height, plant_functional_type, number_of_individuals. Each
    #     row in the csv should represent one cohort.

    #     :param cell_area: the area of the cell at each location, this is assumed to be
    #     the same across all the locations in the csv.
    #     :param csv_path: path to the csv containing community data, as detailed above.
    #     :param flora: a flora object, ie a dictionary of plant functional properties,
    #     keyed by pft name.
    #     :return: a list of community objects, loaded from the csv
    #     file.
    #     """
    #     community_data = pd.read_csv(csv_path)

    #     data_grouped_by_community = community_data.groupby(community_data.cell_id)

    #     communities = []

    #     for cell_id in data_grouped_by_community.groups:
    #         community_dataframe = data_grouped_by_community.get_group(cell_id)
    #         dbh_values = community_dataframe["diameter_at_breast_height"].to_numpy(
    #             dtype=np.float32
    #         )
    #         number_of_individuals = community_dataframe[
    #             "number_of_individuals"
    #         ].to_numpy(dtype=np.int_)
    #         pft_names = community_dataframe["plant_functional_type"].to_numpy(
    #               dtype=str
    #         )
    #         community_object = Community(
    #             cell_id,  # type:ignore
    #             cell_area,
    #             dbh_values,
    #             number_of_individuals,
    #             pft_names,
    #             flora,
    #         )
    #         communities.append(community_object)

    #     return communities
