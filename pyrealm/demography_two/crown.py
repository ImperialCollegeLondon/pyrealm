"""A set of functions implementing the crown shape and vertical leaf distribution model
used in PlantFATE :cite:t:`joshi:2022a`.
"""  # noqa: D205

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from pyrealm.core.experimental import warn_experimental
from pyrealm.demography_two.cohorts import Cohorts
from pyrealm.demography_two.core import ToDataFrameMixin
from pyrealm.demography_two.tmodel import StemAllometry


def calculate_relative_crown_radius_at_z(
    z: NDArray[np.floating],
    stem_height: NDArray[np.floating],
    m: NDArray[np.floating],
    n: NDArray[np.floating],
    clip: bool = True,
) -> NDArray[np.floating]:
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
        stem_height: Total height of individual stems
        m: Canopy shape parameter of PFT for stems
        n: Canopy shape parameter of PFT for stems
        clip: Boolean flag to set :math:`q(z) = 0` where the :math:`z` is below zero or
            above the stem height.
    """

    z_over_height = z / stem_height
    q_z = m * n * z_over_height ** (n - 1) * (1 - z_over_height**n) ** (m - 1)

    # Set predictions to zero where z is below zero or above the stem height.
    if clip:
        q_z = np.where(np.logical_and(z >= 0, z <= stem_height), q_z, 0)

    return q_z


def calculate_crown_radius(
    q_z: NDArray[np.floating],
    r0: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    # TODO - think about validation here. qz must be row array or 2D (N, n_pft)

    return r0 * q_z


def calculate_stem_projected_crown_area_at_z(
    z: NDArray[np.floating],
    q_z: NDArray[np.floating],
    stem_height: NDArray[np.floating],
    crown_area: NDArray[np.floating],
    q_m: NDArray[np.floating],
    z_max: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    # Calculate A_p
    # Calculate Ap given z > zm
    A_p = crown_area * (q_z / q_m) ** 2
    # Set Ap = Ac where z <= zm
    A_p = np.where(z <= z_max, crown_area, A_p)
    # Set Ap = 0 where z > H
    A_p = np.where(z > stem_height, 0, A_p)

    return A_p


def calculate_stem_projected_leaf_area_at_z(
    z: NDArray[np.floating],
    q_z: NDArray[np.floating],
    stem_height: NDArray[np.floating],
    crown_area: NDArray[np.floating],
    f_g: NDArray[np.floating],
    q_m: NDArray[np.floating],
    z_max: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    # NOTE: Although the internals of this function overlap a lot with
    #       calculate_stem_projected_crown_area_at_z, we want that function to be as
    #       lean as possible, as it used within solve_community_projected_crown_area.

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


class CrownProfile(ToDataFrameMixin):
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
    for calculating canopy models, but can be useful for exploring the behaviour of
    projected area on the same linear scale as the crown radius.

    Args:
        cohorts: A cohorts instance.
        z: An array of vertical height values at which to calculate crown profiles.
        allometry: A StemAllometry instance for the provided cohorts. If this is
            missing, then it is calculated automatically from the cohorts.
    """

    __experimental__ = True

    _array_attrs: ClassVar[tuple[str, ...]] = (
        "cohort_ids",
        "z",
        "relative_crown_radius",
        "crown_radius",
        "projected_crown_area",
        "projected_leaf_area",
        "projected_crown_radius",
        "projected_leaf_radius",
    )

    # Crown profiles are _always_ 2D
    _ndims = 2

    def __init__(
        self,
        cohorts: Cohorts,
        z: NDArray[np.floating],
        allometry: StemAllometry | None = None,
    ):
        """Populate crown profile attributes from the traits, allometry and height."""

        warn_experimental("CrownProfile")

        # TODO: Need to do some form of validation here to check allometry and cohorts
        #       are congruent and sensible - maybe calculate allometry internally.
        #       This is only intended to work with the specific cohort DBH, so could
        #       just check cohort_ids align across the two inputs.

        self.relative_crown_radius: NDArray[np.floating]
        """A 2D array of the relative crown radius of each stem at given heights"""
        self.crown_radius: NDArray[np.floating]
        """A 2D array of the actual crown radius of each stem at given heights"""
        self.projected_crown_area: NDArray[np.floating]
        """A 2D array of the projected crown area of each stem at given heights"""
        self.projected_leaf_area: NDArray[np.floating]
        """A 2D array of the projected leaf area of each stem at given heights"""
        self.stem_height: NDArray[np.floating]
        """A 1D array of the stem heights for each cohort."""

        self.height_is_valid: NDArray[np.bool]
        """A 2D logical array showing which heights are below the stem height for each
        stem."""

        cohort_ids: NDArray = cohorts.cohorts["cohort_id"].to_numpy()

        # Handle allometry
        if allometry is None:
            allometry = StemAllometry(cohorts=cohorts)
        else:
            # Check the allometry is 1D and matches the cohorts
            if allometry._ndims > 1:
                raise ValueError("Provided allometry calculated using `at_dbh`.")
            if not np.all(np.equal(allometry.cohort_ids, cohort_ids)):
                raise ValueError("Provided allometry does not match cohorts.")

        # Validate z and set height_is_valid
        if z.ndim != 1 or np.any(z < 0) or not (np.all(np.diff(z)) > 0):
            raise ValueError(
                "The z value must be a one dimensional array of increasing heights "
                "greater than or equal to 0."
            )

        # Rotate z into a column array and broadcast to prediction shape
        prediction_shape = (len(z), cohorts.n_cohorts)
        self.z: NDArray[np.floating] = np.broadcast_to(z[:, None], prediction_shape)
        """Heights of crown profile predictions."""

        self.cohort_ids = np.broadcast_to(cohort_ids, prediction_shape)
        """Cohort ids."""

        # Store cohort x height array showing which heights are <= stem height.
        self.stem_height = allometry.stem_height
        self.height_is_valid = np.less_equal(self.z, allometry.stem_height)

        # Calculate relative crown radius
        self.relative_crown_radius = calculate_relative_crown_radius_at_z(
            z=self.z,
            m=cohorts.cohorts["m"].to_numpy(),
            n=cohorts.cohorts["n"].to_numpy(),
            stem_height=allometry.stem_height,
        )

        # Calculate actual radius
        self.crown_radius = calculate_crown_radius(
            q_z=self.relative_crown_radius,
            r0=allometry.crown_r0,
        )

        # Calculate projected crown area
        self.projected_crown_area = calculate_stem_projected_crown_area_at_z(
            z=self.z,
            q_z=self.relative_crown_radius,
            crown_area=allometry.crown_area,
            q_m=cohorts.cohorts["q_m"].to_numpy(),
            stem_height=allometry.stem_height,
            z_max=allometry.crown_z_max,
        )

        # Calculate projected leaf area
        self.projected_leaf_area = calculate_stem_projected_leaf_area_at_z(
            z=self.z,
            q_z=self.relative_crown_radius,
            f_g=cohorts.cohorts["f_g"].to_numpy(),
            q_m=cohorts.cohorts["q_m"].to_numpy(),
            crown_area=allometry.crown_area,
            stem_height=allometry.stem_height,
            z_max=allometry.crown_z_max,
        )

    def __repr__(self) -> str:
        return "CrownProfile: Prediction for {1} cohorts at {0} heights.".format(
            *self.relative_crown_radius.shape
        )

    @property
    def projected_crown_radius(self) -> NDArray[np.floating]:
        """An array of the projected crown radius of stems at z heights."""
        return np.sqrt(self.projected_crown_area / np.pi)

    @property
    def projected_leaf_radius(self) -> NDArray[np.floating]:
        """An array of the projected leaf radius of stems at z heights."""
        return np.sqrt(self.projected_leaf_area / np.pi)

    def to_xy(
        self,
        attr: str,
        stem_offsets: NDArray[np.floating] | None = None,
        two_sided: bool = True,
        as_xy: bool = False,
    ) -> (
        list[tuple[NDArray[np.floating], NDArray[np.floating]]]
        | list[NDArray[np.floating]]
    ):
        """Extract plotting data from crown profiles.

        A CrownProfile instance contains crown radius and projected area data for a set
        of stems at given heights, but can contain predictions of these attributes above
        the actual heights of some or all of the stems.

        This function extracts plotting data for a given attribute for each crown that
        includes only the predictions within the height range of the actual stem. It can
        also mirror the values around the vertical midline to provide a two sided canopy
        shape.

        The data are returned as a list with one entry per stem. The default value for
        each entry a tuple of two arrays (height, attribute values) but the `as_xy=True`
        option will return an `(N, 2)` dimensioned XY array suitable for use with
        :class:`matplotlib.patches.Polygon`.

        Args:
            attr: The crown profile attribute to plot (see
                :class:`~pyrealm.demography.crown.CrownProfile`)
            stem_offsets: An optional array of offsets to add to the midline of stems.
            two_sided: Should the plotting data show a two sided canopy.
            as_xy: Should the plotting data be returned as a single XY array rather than
                tuples of X and Y coordinates.
        """

        # Input validation
        if attr not in self._array_attrs:
            raise ValueError(f"Unknown crown profile attribute: {attr}")

        # Get the attributes, setting above height values to NaN and broadcast the
        # heights to match
        vals = np.where(self.height_is_valid, getattr(self, attr), np.nan)
        z = np.where(self.height_is_valid, np.broadcast_to(self.z, vals.shape), np.nan)

        # Get the plotting coordinates
        if two_sided:
            attr_stack = np.concatenate(
                [vals, np.zeros_like(self.stem_height)[None, :], np.flipud(vals)]
            )
            z_stack = np.concatenate([z, self.stem_height[None, :], np.flipud(z)])
        else:
            attr_stack = np.concatenate(
                [vals, np.zeros_like(self.stem_height)[None, :]]
            )
            z_stack = np.concatenate([z, self.stem_height[None, :]])

        # Add stem offsets if provided
        if stem_offsets is not None:
            attr_stack += stem_offsets

        # Strip out NaN and get per stem plotting
        data: (
            list[tuple[NDArray[np.floating], NDArray[np.floating]]]
            | list[NDArray[np.floating]]
        ) = []

        for cht_idx in np.arange(self.stem_height.size):
            # Get the indices of the non NaN values
            not_nan = ~np.isnan(attr_stack[:, cht_idx])

            if as_xy:
                # Combine the values into an (N,2) XY array and drop nans
                data.append(
                    np.hstack([attr_stack[:, [cht_idx]], z_stack[:, [cht_idx]]])[
                        not_nan
                    ]
                )
            else:
                # Return the individual 1D arrays, dropping nans`
                # Unclear why mypy refuses to recognise this as a two tuple of NDArrays
                data.append((z_stack[not_nan, cht_idx], attr_stack[not_nan, cht_idx]))  # type: ignore

        return data
