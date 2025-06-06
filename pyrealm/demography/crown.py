"""A set of functions implementing the crown shape and vertical leaf distribution model
used in PlantFATE :cite:t:`joshi:2022a`.
"""  # noqa: D205

from dataclasses import InitVar, dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from pyrealm.core.experimental import warn_experimental
from pyrealm.demography.core import (
    PandasExporter,
    _validate_demography_array_arguments,
)
from pyrealm.demography.flora import Flora, StemTraits
from pyrealm.demography.tmodel import StemAllometry


def calculate_relative_crown_radius_at_z(
    z: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    m: NDArray[np.float64],
    n: NDArray[np.float64],
    validate: bool = True,
    clip: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate relative crown radius at a given height.

    The crown shape parameters ``m`` and ``n`` define the vertical distribution of
    crown along the stem. For a stem of a given total height, this function calculates
    the relative crown radius at a given height :math:`z`:

    .. math::

        q(z) = m n \left(\dfrac{z}{H}\right) ^ {n -1}
        \left( 1 - \left(\dfrac{z}{H}\right) ^ n \right)^{m-1}

    This function calculates :math:`q(z)` across a set of stems: the ``stem_height``,
    ``m`` and ``n`` arguments should be one-dimensional arrays ('row vectors') of equal
    length :math:`I`.  The value for ``z`` is then an array of heights, with one of the
    following shapes:

    1. A scalar array: :math:`q(z)` is found for all stems at the same height and the
       return value is a 1D array of length :math:`I`.
    2. A row vector of length :math:`I`: :math:`q(z)` is found for all stems at
       stem-specific heights and the return value is again a 1D array of length
       :math:`I`.
    3. A column vector of length :math:`J`, that is a 2 dimensional array of shape
       (:math:`J`, 1). This allows :math:`q(z)` to be calculated efficiently for a set
       of heights for all stems and return a 2D array of shape (:math:`J`, :math:`I`).

    By default, this function clips :math:`q(z)`: the value is set to zero for values of
    :math:`z < 0` or :math:`z > H`.

    Args:
        z: Height at which to calculate relative radius
        stem_height: Total height of individual stem
        m: Canopy shape parameter of PFT
        n: Canopy shape parameter of PFT
        validate: Boolean flag to suppress argument validation.
        clip: Boolean flag to set :math:`q(z) = 0` where the :math:`z` is below zero or
            above the stem height.
    """

    if validate:
        _validate_demography_array_arguments(
            trait_args={"m": m, "n": n}, size_args={"stem_height": stem_height, "z": z}
        )

    z_over_height = z / stem_height
    q_z = m * n * z_over_height ** (n - 1) * (1 - z_over_height**n) ** (m - 1)

    # Set predictions to zero where z is below zero or above the stem height.
    if clip:
        q_z = np.where(np.logical_and(z >= 0, z <= stem_height), q_z, 0)

    return q_z


def calculate_crown_radius(
    q_z: NDArray[np.float64],
    r0: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate crown radius from relative crown radius and crown r0.

    The relative crown radius (:math:`q(z)`) at a given height :math:`z` describes the
    vertical profile of the crown shape, but only varies with the ``m`` and ``n`` shape
    parameters and the stem height. The actual crown radius at a given height
    (:math:`r(z)`) needs to be scaled using :math:`r_0` such that the maximum crown area
    equals the expected crown area given the crown area ratio traiit for the plant
    functional type:

    .. math::

        r(z) = r_0 q(z)

    This function calculates :math:`r(z)` given estimated ``r0`` and an array of
    relative radius values.

    Args:
        q_z: An array of relative crown radius values
        r0:  An array of crown radius scaling factor values
        validate: Boolean flag to suppress argument validation.
    """

    # TODO - think about validation here. qz must be row array or 2D (N, n_pft)

    return r0 * q_z


def calculate_stem_projected_crown_area_at_z(
    z: NDArray[np.float64],
    q_z: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    crown_area: NDArray[np.float64],
    q_m: NDArray[np.float64],
    z_max: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    """Calculate stem projected crown area above a given height.

    This function calculates the projected crown area of a set of stems with given
    properties at a set of vertical heights. The stem properties are given in the
    arguments ``stem_height``,``crown_area``,``q_m`` and ``z_max``, which must be
    one-dimensional arrays ('row vectors') of equal length. The array of vertical
    heights ``z`` accepts a range of input shapes (see
    :meth:`~pyrealm.demography.crown.calculate_relative_crown_radius_at_z`
    ) and this function then also requires the expected relative stem radius (``q_z``)
    calculated from those heights.

    Args:
        z: Vertical height at which to estimate crown area
        q_z: Relative crown radius at those heights
        crown_area: Crown area of each stem
        stem_height: Stem height of each stem
        q_m: Canopy shape parameter ``q_m``` for each stem
        z_max: Height of maximum crown radius for each stem
        validate: Boolean flag to suppress argument validation.
    """

    if validate:
        _validate_demography_array_arguments(
            trait_args={"q_m": q_m},
            size_args={
                "stem_height": stem_height,
                "crown_area": crown_area,
                "z": z,
                "z_max": z_max,
            },
            at_size_args={"q_z": q_z},
        )

    # Calculate A_p
    # Calculate Ap given z > zm
    A_p = crown_area * (q_z / q_m) ** 2
    # Set Ap = Ac where z <= zm
    A_p = np.where(z <= z_max, crown_area, A_p)
    # Set Ap = 0 where z > H
    A_p = np.where(z > stem_height, 0, A_p)

    return A_p


def calculate_stem_projected_leaf_area_at_z(
    z: NDArray[np.float64],
    q_z: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    crown_area: NDArray[np.float64],
    f_g: NDArray[np.float64],
    q_m: NDArray[np.float64],
    z_max: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    """Calculate projected leaf area above a given height.

    This function calculates the projected leaf area of a set of stems with given
    properties at a set of vertical heights. This differs from crown area in allowing
    for crown openness within the crown of an individual stem that results in the
    displacement of leaf area further down into the crown. The degree of openness is
    controlled by the crown gap fraction property of each stem.

    The stem properties are given in the arguments
    ``stem_height``,``crown_area``,``f_g``,``q_m`` and ``z_max``, which must be
    one-dimensional arrays ('row vectors') of equal length. The array of vertical
    heights ``z`` accepts a range of input shapes (see
    :meth:`~pyrealm.demography.crown.calculate_relative_crown_radius_at_z`
    ) and this function then also requires the expected relative stem radius (``q_z``)
    calculated from those heights.

    Args:
        z: Vertical heights on the z axis.
        q_z: Relative crown radius at heights in z.
        crown_area: Crown area for a stem
        stem_height: Total height of a stem
        f_g: Within crown gap fraction for each stem.
        q_m: Canopy shape parameter ``q_m``` for each stem
        z_max: Height of maximum crown radius for each stem
        validate: Boolean flag to suppress argument validation.
    """

    # NOTE: Although the internals of this function overlap a lot with
    #       calculate_stem_projected_crown_area_at_z, we want that function to be as
    #       lean as possible, as it used within solve_community_projected_crown_area.

    if validate:
        _validate_demography_array_arguments(
            trait_args={"q_m": q_m, "f_g": f_g},
            size_args={
                "stem_height": stem_height,
                "crown_area": crown_area,
                "z": z,
                "z_max": z_max,
            },
            at_size_args={"q_z": q_z},
        )

    # Calculate Ac terms
    A_c_terms = crown_area * (q_z / q_m) ** 2

    # Set Acp either side of z_max
    A_cp = np.where(
        z <= z_max,
        crown_area - A_c_terms * f_g,
        A_c_terms * (1 - f_g),
    )
    # Set Ap = 0 where z > H
    A_cp = np.where(z > stem_height, 0, A_cp)

    return A_cp


@dataclass
class CrownProfile(PandasExporter):
    """Calculate vertical crown profiles for stems.

    This method calculates crown profile predictions, given an array of vertical
    heights (``z``) for:

    * relative crown radius,
    * actual crown radius,
    * projected crown area, and
    * projected leaf area.

    The predictions require a set of plant functional types (PFTs) but also the expected
    allometric predictions of stem height, crown area and z_max for an actual stem of a
    given size for each PFT.

    In addition to the variables above, the class can also has properties the calculate
    the projected crown radius and projected leaf radius. These are simply the radii
    that would result in the two projected areas: the values are not directly meaningful
    for calculating canopy models, but can be useful for exploring the behavour of
    projected area on the same linear scale as the crown radius.

    Args:
        stem_traits: A Flora or StemTraits instance providing plant functional trait
            data.
        stem_allometry: A StemAllometry instance setting the stem allometries for the
            crown profile.
        z: An array of vertical height values at which to calculate crown profiles.
        validate: Boolean flag to suppress argument validation.
    """

    array_attrs: ClassVar[tuple[str, ...]] = (
        "relative_crown_radius",
        "crown_radius",
        "projected_crown_area",
        "projected_leaf_area",
        "projected_crown_radius",
        "projected_leaf_radius",
    )

    stem_traits: InitVar[StemTraits | Flora]
    """A Flora or StemTraits instance providing plant functional trait data."""
    stem_allometry: InitVar[StemAllometry]
    """A StemAllometry instance setting the stem allometries for the crown profile."""
    z: NDArray[np.float64]
    """An array of vertical height values at which to calculate crown profiles."""
    validate: InitVar[bool] = True
    """Boolean flag to suppress argument validation."""

    relative_crown_radius: NDArray[np.float64] = field(init=False)
    """An array of the relative crown radius of stems at z heights"""
    crown_radius: NDArray[np.float64] = field(init=False)
    """An array of the actual crown radius of stems at z heights"""
    projected_crown_area: NDArray[np.float64] = field(init=False)
    """An array of the projected crown area of stems at z heights"""
    projected_leaf_area: NDArray[np.float64] = field(init=False)
    """An array of the projected leaf area of stems at z heights"""

    # Information attributes
    _n_pred: int = field(init=False)
    """The number of predictions per stem."""
    _n_stems: int = field(init=False)
    """The number of stems."""

    __experimental__ = True

    def __post_init__(
        self,
        stem_traits: StemTraits | Flora,
        stem_allometry: StemAllometry,
        validate: bool,
    ) -> None:
        """Populate crown profile attributes from the traits, allometry and height."""

        warn_experimental("CrownProfile")

        # If validation is required, only need to perform validation once to check that
        # the at_dbh values are congruent with the stem_traits inputs. If they are, then
        # all the other allometry function inputs will be too.
        if validate:
            _validate_demography_array_arguments(
                trait_args={"h_max": stem_traits.h_max}, size_args={"z": self.z}
            )

        # Calculate relative crown radius
        self.relative_crown_radius = calculate_relative_crown_radius_at_z(
            z=self.z,
            m=stem_traits.m,
            n=stem_traits.n,
            stem_height=stem_allometry.stem_height,
            validate=False,
        )

        # Calculate actual radius
        self.crown_radius = calculate_crown_radius(
            q_z=self.relative_crown_radius, r0=stem_allometry.crown_r0, validate=False
        )

        # Calculate projected crown area
        self.projected_crown_area = calculate_stem_projected_crown_area_at_z(
            z=self.z,
            q_z=self.relative_crown_radius,
            crown_area=stem_allometry.crown_area,
            q_m=stem_traits.q_m,
            stem_height=stem_allometry.stem_height,
            z_max=stem_allometry.crown_z_max,
            validate=False,
        )

        # Calculate projected leaf area
        self.projected_leaf_area = calculate_stem_projected_leaf_area_at_z(
            z=self.z,
            q_z=self.relative_crown_radius,
            f_g=stem_traits.f_g,
            q_m=stem_traits.q_m,
            crown_area=stem_allometry.crown_area,
            stem_height=stem_allometry.stem_height,
            z_max=stem_allometry.crown_z_max,
            validate=False,
        )

        # Set the number of observations per stem (one if dbh is 1D, otherwise size of
        # the first axis)
        if self.relative_crown_radius.ndim == 1:
            self._n_pred = 1
        else:
            self._n_pred = self.relative_crown_radius.shape[0]

        self._n_stems = stem_traits._n_stems

    def __repr__(self) -> str:
        return (
            f"CrownProfile: Prediction for {self._n_stems} stems "
            f"at {self._n_pred} observations."
        )

    @property
    def projected_crown_radius(self) -> NDArray[np.float32]:
        """An array of the projected crown radius of stems at z heights."""
        return np.sqrt(self.projected_crown_area / np.pi)

    @property
    def projected_leaf_radius(self) -> NDArray[np.float32]:
        """An array of the projected leaf radius of stems at z heights."""
        return np.sqrt(self.projected_leaf_area / np.pi)


def get_crown_xy(
    crown_profile: CrownProfile,
    stem_allometry: StemAllometry,
    attr: str,
    stem_offsets: NDArray[np.float32] | None = None,
    two_sided: bool = True,
    as_xy: bool = False,
) -> list[tuple[NDArray, NDArray]] | list[NDArray]:
    """Extract plotting data from crown profiles.

    A CrownProfile instance contains crown radius and projected area data for a set of
    stems at given heights, but can contain predictions of these attributes above the
    actual heights of some or all of the stems or indeed below ground.

    This function extracts plotting data for a given attribute for each crown that
    includes only the predictions within the height range of the actual stem. It can
    also mirror the values around the vertical midline to provide a two sided canopy
    shape.

    The data are returned as a list with one entry per stem. The default value for each
    entry a tuple of two arrays (height, attribute values) but the `as_xy=True` option
    will return an `(N, 2)` dimensioned XY array suitable for use with
    {class}`~matplotlib.patches.Polygon`.

    Args:
        crown_profile: A crown profile instance
        stem_allometry: The stem allometry instance used to create the crown profile
        attr: The crown profile attribute to plot (see
            :class:`~pyrealm.demography.crown.CrownProfile`)
        stem_offsets: An optional array of offsets to add to the midline of stems.
        two_sided: Should the plotting data show a two sided canopy.
        as_xy: Should the plotting data be returned as a single XY array.

    """

    # Input validation
    if attr not in crown_profile.array_attrs:
        raise ValueError(f"Unknown crown profile attribute: {attr}")

    # Get the attribute and flatten the heights from a column array to one dimensional
    attr_values = getattr(crown_profile, attr)
    z = crown_profile.z.flatten()

    # Orient the data so that lower heights always come first
    if z[0] < z[-1]:
        z = np.flip(z)
        attr_values = np.flip(attr_values, axis=0)

    # Collect the per stem data
    crown_plotting_data: list = []

    for stem_index in np.arange(attr_values.shape[1]):
        # Find the heights and values that fall within the individual stem
        height_is_valid = np.logical_and(
            z <= stem_allometry.stem_height[:, stem_index], z >= 0
        )
        valid_attr_values: NDArray = attr_values[height_is_valid, stem_index]
        valid_heights: NDArray = z[height_is_valid]

        if two_sided:
            # The values are extended to include the reverse profile as well as the zero
            # value at the stem height
            valid_heights = np.concatenate(
                [
                    np.flip(valid_heights),
                    stem_allometry.stem_height[:, stem_index],
                    valid_heights,
                ]
            )
            valid_attr_values = np.concatenate(
                [-np.flip(valid_attr_values), [0], valid_attr_values]
            )
        else:
            # Only the zero value is added
            valid_heights = np.concatenate(
                [
                    stem_allometry.stem_height[:, stem_index],
                    valid_heights,
                ]
            )
            valid_attr_values = np.concatenate([[0], valid_attr_values])

        # Add offsets if provided
        if stem_offsets is not None:
            valid_attr_values += stem_offsets[stem_index]

        if as_xy:
            # Combine the values into an (N,2) XY array
            crown_plotting_data.append(
                np.hstack([valid_attr_values[:, None], valid_heights[:, None]])
            )
        else:
            # Return the individual 1D arrays
            crown_plotting_data.append((valid_heights, valid_attr_values))

    return crown_plotting_data
