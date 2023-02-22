"""The :mod:`~pyrealm.tmodel` module provides an implementation of the T Model of plant
growth given an estimate of gross primary productivity (GPP).

* The growth form and productivity allocation model of a plant is set using
    :class:`~pyrealm.constants.tmodel_const.TModelTraits`.
* The class :class:`~pyrealm.tmodel.TTree` is used to generate an instance of a
  plant to be simulated, with methods :meth:`~pyrealm.tmodel.TTree.reset_diameters`
  and :meth:`~pyrealm.tmodel.TTree.calculate_growth` to calculate the plant
  geometry for a given diameter or predict growth from estimated GPP.
* The function :func:`~pyrealm.tmodel.grow_ttree` predicts plant growth through
  time given a time series of GPP.
"""  # noqa: D205, D415

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants.tmodel_const import TModelTraits

# Design Notes:
#
# One functionally easy thing to do the TTree object is to expose the geometry
# methods (e.g. by having TTree.calculate_crown_area()) to allow users to
# subclass the object and substitute their own geometry. Which is neat but the
# geometry and growth are not separate and there is no guarantee that a
# user-supplied geometry behaves as expected. So, easy to implement and a
# natural solution to people wanting to tinker with the geometry but not going
# there now.


class TTree:
    """Model plant growth using the T model.

    This class provides an implementation of the calculations of tree geometry, mass and
    growth described by :cite:t:`Li:2014bc`. All of the properties of the T model are
    derived from a set of traits (see
    :class:`~pyrealm.constants.tmodel_const.TModelTraits`), stem diameter measurements
    and estimates of gross primary productivity.

    See the details of :meth:`~pyrealm.tmodel.TTree.reset_diameters` and
    :meth:`~pyrealm.tmodel.TTree.calculate_growth` for details of the properties and
    calculations.

    Args:
        traits: An object of class :class:`~pyrealm.constants.tmodel_const.TModelTraits`
        diameters: A float or np.array of stem diameters.
    """

    def __init__(
        self,
        diameters: NDArray,
        traits: TModelTraits = TModelTraits(),
    ) -> None:
        self.traits: TModelTraits = traits

        # The diameter is used to define all of the geometric scaling
        # based on the trait parameters.

        self._diameter: NDArray
        self._height: NDArray
        self._crown_fraction: NDArray
        self._crown_area: NDArray
        self._mass_stm: NDArray
        self._mass_fol: NDArray
        self._mass_swd: NDArray

        self.reset_diameters(diameters)

        # Growth is then applied by providing estimated gpp using the
        # calculate_growth() method, which populates the following:
        self.growth_calculated: bool = False
        self._gpp_raw: NDArray
        self._gpp_actual: NDArray
        self._npp: NDArray
        self._resp_swd: NDArray
        self._resp_frt: NDArray
        self._resp_fol: NDArray
        self._turnover: NDArray
        self._d_mass_s: NDArray
        self._d_mass_fr: NDArray
        self._delta_d: NDArray
        self._delta_mass_stm: NDArray
        self._delta_mass_frt: NDArray

    def _check_growth_calculated(self, property: str) -> NDArray:
        """Helper function to return growth values if calculated.

        This acts as a gatekeeper to make sure that a growth property is not returned
        before calculate_growth() has been run on the current diameters.

        Args:
            property: The property value to return if available.
        """
        if not self.growth_calculated:
            raise RuntimeError("Growth estimates not calculated: use calculate_growth")

        return getattr(self, property)

    @property
    def diameter(self) -> NDArray:
        """Individual diameter (m)."""  # noqa: D402
        return self._diameter

    @property
    def height(self) -> NDArray:
        """Individual height (m)."""  # noqa: D402
        return self._height

    @property
    def crown_fraction(self) -> NDArray:
        """Individual crown fraction (unitless)."""
        return self._crown_fraction

    @property
    def crown_area(self) -> NDArray:
        """Individual crown area (m2)."""
        return self._crown_area

    @property
    def mass_swd(self) -> NDArray:
        """Individual softwood mass (kg)."""
        return self._mass_swd

    @property
    def mass_stm(self) -> NDArray:
        """Individual stem mass (kg)."""
        return self._mass_stm

    @property
    def mass_fol(self) -> NDArray:
        """Individual foliage mass (kg)."""
        return self._mass_fol

    @property
    def gpp_raw(self) -> NDArray:
        """Raw gross primary productivity."""
        return self._check_growth_calculated("_gpp_raw")

    @property
    def gpp_actual(self) -> NDArray:
        """Actual gross primary productivity."""
        return self._check_growth_calculated("_gpp_actual")

    @property
    def resp_swd(self) -> NDArray:
        """Individual softwood respiration costs."""
        return self._check_growth_calculated("_resp_swd")

    @property
    def resp_frt(self) -> NDArray:
        """Individual fine root respiration costs."""
        return self._check_growth_calculated("_resp_frt")

    @property
    def resp_fol(self) -> NDArray:
        """Individual foliar respiration costs."""
        return self._check_growth_calculated("_resp_fol")

    @property
    def npp(self) -> NDArray:
        """Net primary productivity."""
        return self._check_growth_calculated("_npp")

    @property
    def turnover(self) -> NDArray:
        """Plant turnover."""
        return self._check_growth_calculated("_turnover")

    @property
    def d_mass_s(self) -> NDArray:
        """Individual relative change in mass."""
        return self._check_growth_calculated("_d_mass_s")

    @property
    def d_mass_fr(self) -> NDArray:
        """Individual relative change in fine root mass."""
        return self._check_growth_calculated("_d_mass_fr")

    @property
    def delta_d(self) -> NDArray:
        """Individual change in diameter."""
        return self._check_growth_calculated("_delta_d")

    @property
    def delta_mass_stm(self) -> NDArray:
        """Individual total change in stem mass."""
        return self._check_growth_calculated("_delta_mass_stm")

    @property
    def delta_mass_frt(self) -> NDArray:
        """Individual total change in fine root mass."""
        return self._check_growth_calculated("_delta_mass_frt")

    def reset_diameters(self, values: NDArray) -> None:
        """Reset the stem diameters for the T model.

        The set_diameter method can be used to reset the diameter values and then uses
        these values to populate the geometric and mass properties that scale with stem
        diameter.

        * Height (m, ``height``, :math:`H`):

        .. math::

            H = H_{max} ( 1 - e^{a D/ H_{max}}

        TODO: complete this description.
        """

        self._diameter = values

        # Height of tree from diameter, Equation (4) of Li ea.
        self._height = self.traits.h_max * (
            1 - np.exp(-self.traits.a_hd * self._diameter / self.traits.h_max)
        )

        # Crown area of tree, Equation (8) of Li ea.
        self._crown_area = (
            ((np.pi * self.traits.ca_ratio) / (4 * self.traits.a_hd))
            * self._diameter
            * self._height
        )

        # Crown fraction, Equation (11) of Li ea.
        self._crown_fraction = self._height / (self.traits.a_hd * self._diameter)

        # Masses
        self._mass_stm = (
            (np.pi / 8) * (self._diameter**2) * self._height * self.traits.rho_s
        )
        self._mass_fol = self._crown_area * self.traits.lai * (1 / self.traits.sla)
        self._mass_swd = (
            self._crown_area
            * self.traits.rho_s
            * self._height
            * (1 - self._crown_fraction / 2)
            / self.traits.ca_ratio
        )

        # Flag any calculated growth values as outdated
        self.growth_calculated = False

    def calculate_growth(self, gpp: NDArray) -> None:
        """Calculate growth predictions given a GPP estimate.

        This method updates the instance with predicted changes in tree geometry, mass
        and respiration costs from the initial state given an estimate of gross primary
        productivity (GPP).

        Args:
            gpp: Primary productivity
        """

        # GPP fixed per m2 of crown
        self._gpp_raw = gpp
        gpp_unit_cr = self._gpp_raw * (
            1 - np.exp(-(self.traits.par_ext * self.traits.lai))
        )
        self._gpp_actual = self._crown_area * gpp_unit_cr

        # Respiration costs (Eqn 13 of Li ea)
        # - sapwood, fine root and foliage maintenance
        self._resp_swd = self._mass_swd * self.traits.resp_s
        self._resp_frt = (
            self.traits.zeta * self.traits.sla * self._mass_fol * self.traits.resp_r
        )
        self._resp_fol = self._gpp_actual * self.traits.resp_f

        # Net primary productivity
        self._npp = self.traits.yld * (
            self._gpp_actual - self._resp_fol - self._resp_frt - self._resp_swd
        )

        # Turnover costs for foliage and fine roots
        self._turnover = (
            self._crown_area
            * self.traits.lai
            * (
                (1 / (self.traits.sla * self.traits.tau_f))
                + (self.traits.zeta / self.traits.tau_r)
            )
        )

        # relative increments - these are used to calculate delta_d and
        # then scaled by delta_d to give actual increments
        self._d_mass_s = (
            np.pi
            / 8
            * self.traits.rho_s
            * self._diameter
            * (
                self.traits.a_hd
                * self._diameter
                * (1 - (self._height / self.traits.h_max))
                + 2 * self._height
            )
        )

        self._d_mass_fr = (
            self.traits.lai
            * ((np.pi * self.traits.ca_ratio) / (4 * self.traits.a_hd))
            * (
                self.traits.a_hd
                * self._diameter
                * (1 - self._height / self.traits.h_max)
                + self._height
            )
            * (1 / self.traits.sla + self.traits.zeta)
        )

        # Actual increments
        self._delta_d = (self._npp - self._turnover) / (
            self._d_mass_s + self._d_mass_fr
        )
        self._delta_mass_stm = self._d_mass_s * self._delta_d
        self._delta_mass_frt = self._d_mass_fr * self._delta_d

        self.growth_calculated = True


def grow_ttree(
    gpp: NDArray,
    d_init: NDArray,
    time_axis: int,
    traits: TModelTraits = TModelTraits(),
    outvars: tuple[str, ...] = ("diameter", "height", "crown_area", "delta_d"),
) -> dict[str, np.ndarray]:
    """Fit a growth time series using the T Model."""
    raise NotImplementedError()


#     This function fits the T Model incrementally to a set of modelled plants,
#     given a time series of GPP estimates.

#     Args:
#         gpp: An array of GPP values
#         d_init: An array of starting diameters
#         traits: Traits to be used
#         time_axis: An axis in P0 and d that represents an annual time series
#         outvars: A list of Tree properties to store.

#     Returns:
#         A dictionary of estimates for the time series, exporting the values
#         specified in `outvars`
#     """

#     # The gpp array should contain a time axis to increment over. The d_init input
#     # should then be the same shape as P0 _without_ that time axis dimension
#     # and the loop will iterate over the years

#     gpp_shape = gpp.shape
#     if time_axis < 0 or time_axis >= len(gpp_shape):
#         raise RuntimeError(f"time_axis must be >= 0 and <= {len(gpp_shape) - 1}")

#     # Check that the input shapes for a single year match the shape of the
#     # initial diameters
#     single_year_gpp = np.take(gpp, indices=0, axis=time_axis)
#     _ = check_input_shapes(single_year_gpp, d_init)

#     # TODO: - handle 1D GPP time series applied to more than one diameter

#     # Initialise the Tree object
#     tree = TTree(d_init, traits)

#     # Check the requested outvars
#     if "diameter" not in outvars:
#         raise RuntimeWarning("output_vars must include diameter")

#     badvars = []
#     for var in outvars:

#         try:
#             _ = getattr(tree, var)
#         except AttributeError:
#             badvars.append(var)

# if badvars:
#     raise RuntimeError(
#         f"Unknown tree properties in outvars: {', '.join(badvars)}"
#         )

#     # Create an array to store the requested variables by adding a dimension
#     # to the gpp input with length set to the number of variables
#     output = {v: np.zeros(gpp.shape) for v in outvars}

#     # Create an indexing object to insert values into the output. This is
#     # a bit obscure: the inputs have a time axis and an arbitrary number
#     # of other dimensions and the output adds another dimension for the
#     # output variables. So this object creates a slice (:) on _all_ dimensions
#     # and the loop then replaces the time axis and variable axis with integers.
#     output_index = [slice(None)] * gpp.ndim

#     # Loop over the gpp time axis
#     for year in np.arange(gpp_shape[time_axis]):

#         # Calculate the growth based on the current year of gpp
#         tree.calculate_growth(np.take(gpp, indices=year, axis=time_axis))

#         # Store the requested variables into the output arrays
#         for each_var in outvars:

#             # Extract variable values from tree into output - set the last index
#             # (variable dimension) to the variable index and the time axis to the year
#             output_index[time_axis] = year
#             output[each_var][tuple(output_index)] = getattr(tree, each_var)

#         # Now update the tree object
#         tree.reset_diameters(tree.diameter + getattr(tree, "delta_d"))

#     return output
