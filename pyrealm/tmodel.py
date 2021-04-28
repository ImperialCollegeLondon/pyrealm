from typing import Union, Tuple
import numpy as np

from pyrealm.pmodel import check_input_shapes
from pyrealm.param_classes import TModelTraits


class TTree:
    """Implementation of the T model

    This class provides an implementation of the calculations of tree geometry,
    mass and growth described by :cite:`Li:2014bc`. All of the properties of
    the T model are derived from a set of traits (see :class:`~pyrealm.tmodel.Traits`),
    stem diameter measurements and estimates of gross primary productivity.

    See the details of :meth:`~pyrealm.tmodel.TTree.set_diameter` and
    :meth:`~pyrealm.tmodel.TTree.calculate_growth` for details of the properties
    and calculations.

    Args:
        traits: An object of class :class:`~pyrealm.param_classes.TModelTraits`
    """

    def __init__(self, traits: TModelTraits = TModelTraits()):

        self.traits = traits

        # The diameter is used to define all of the geometric scaling
        # based on the trait parameters. It is set by the set_diameter()
        # method, which then populates the other geometric variables
        self._diameter = None
        self._height = None
        self._crown_fraction = None
        self._crown_area = None
        self._mass_stm = None
        self._mass_fol = None
        self._mass_swd = None

        # Growth is then applied by providing estimated gpp using the
        # grow() method, which populates the following:
        self._gpp_actual = None
        self._npp = None
        self._resp_swd = None
        self._resp_frt = None
        self._resp_fol = None
        self._turnover = None
        self._delta_d = None
        self._delta_mass_stm = None
        self._delta_mass_frt = None


    @property
    def diameter(self):
        return self._diameter

    @property
    def height(self):
        return self._height

    @property
    def crown_fraction(self):
        return self._crown_fraction

    @property
    def crown_area(self):
        return self._crown_area

    @property
    def mass_swd(self):
        return self._mass_swd

    @property
    def mass_stm(self):
        return self._mass_stm

    @property
    def mass_fol(self):
        return self._mass_fol

    @property
    def gpp_actual(self):
        return self._gpp_actual

    @property
    def resp_swd(self):
        return self._resp_swd

    @property
    def resp_frt(self):
        return self._resp_frt

    @property
    def resp_fol(self):
        return self._resp_fol

    @property
    def npp(self):
        return self._npp

    @property
    def turnover(self):
        return self._turnover

    @property
    def delta_d(self):
        return self._delta_d

    @property
    def delta_mass_stm(self):
        return self._delta_mass_stm

    @property
    def delta_mass_frt(self):
        return self._delta_mass_frt

    def set_diameter(self, values: Union[float, np.ndarray]):
        """Set stem diameter for the T model

        The set_diameter method sets the diameter values and then uses these
        values to populate a geometric and mass properties that scale with
        stem diameter.

        * Height (m, ``height``, :math:`H`):

        .. math::

            H = H_{max} ( 1 - e^{a D/ H_{max}}

        Returns:

        """

        self._diameter = values

        # Height of tree from diameter, Equation (4) of Li ea.
        self._height = (self.traits.h_max *
                        (1 - np.exp(-self.traits.a_hd * self.diameter
                                    / self.traits.h_max)))

        # Crown area of tree, Equation (8) of Li ea.
        self._crown_area = (((np.pi * self.traits.ca_ratio)/(4 * self.traits.a_hd))
                            * self.diameter * self.height)

        # Crown fraction, Equation (11) of Li ea.
        self._crown_fraction = self.height / (self.traits.a_hd * self.diameter)

        # Masses
        self._mass_stm = ((np.pi / 8) * (self.diameter ** 2) *
                          self.height * self.traits.rho_s)
        self._mass_fol = (self.crown_area * self.traits.lai *
                          (1 / self.traits.sla))
        self._mass_swd = (self.crown_area * self.traits.rho_s * self.height *
                          (1 - self.crown_fraction / 2) / self.traits.ca_ratio)

        # Clear any calculated growth values
        self._gpp_actual = None
        self._npp = None
        self._resp_swd = None
        self._resp_frt = None
        self._resp_fol = None
        self._turnover = None
        self._delta_d = None
        self._delta_mass_stm = None
        self._delta_mass_frt = None

    def calculate_growth(self, gpp):
        """
        Grows a tree given estimated primary productivity.

        Args:
            gpp: Primary productivity

        Returns:

        """

        # GPP fixed per m2 of crown
        gpp_unit_cr = gpp * (1 - np.exp(-(self.traits.par_ext * self.traits.lai)))
        self._gpp_actual = self.crown_area * gpp_unit_cr

        # Respiration costs (Eqn 13 of Li ea)
        # - sapwood, fine root and foliage maintenance
        self._resp_swd = self.mass_swd * self.traits.resp_s
        self._resp_frt = (self.traits.zeta * self.traits.sla *
                          self.mass_fol * self.traits.resp_r)
        self._resp_fol = (self.gpp_actual * self.traits.resp_f)

        # Net primary productivity
        self._npp = self.traits.yld * (self.gpp_actual - self.resp_fol -
                                       self.resp_frt - self._resp_swd)

        # Turnover costs for foliage and fine roots
        self._turnover = (self.crown_area * self.traits.lai *
                          ((1 / (self.traits.sla * self.traits.tau_f)) +
                           (self.traits.zeta / self.traits.tau_r)))

        # relative increments - these are used to calculate delta_d and
        # then scaled by delta_d to give actual increments
        d_mass_s = (np.pi / 8 * self.traits.rho_s * self.diameter *
                    (self.traits.a_hd * self.diameter *
                     (1 - (self.height / self.traits.h_max)) + 2 * self.height))

        d_mass_fr = (self.traits.lai *
                     ((np.pi * self.traits.ca_ratio) / (4 * self.traits.a_hd)) *
                     (self.traits.a_hd * self.diameter *
                      (1 - self.height / self.traits.h_max) + self.height) *
                     (1 / self.traits.sla + self.traits.zeta))

        # Actual increments
        self._delta_d = (self.npp - self.turnover) / (d_mass_s + d_mass_fr)
        self._delta_mass_stm = d_mass_s * self.delta_d
        self._delta_mass_frt = d_mass_fr * self.delta_d


def grow_ttree(gpp: Union[float, np.ndarray], d_init: Union[float, np.ndarray],
               time_axis: int, traits: TModelTraits = TModelTraits(),
               outvars: Tuple[str, ...] = ('diameter', 'height', 'crown_area', 'delta_d')):

    """Fits the T Model given a GPP

    Args:
        gpp: An array of GPP values
        d_init: An array of starting diameters
        traits: Traits to be used
        time_axis: An axis in P0 and d that represents an annual time series
        outvars: A list of Tree properties to store.

    Returns:

    """

    # The gpp array should contain a time axis to increment over. The d_init input
    # should then be the same shape as P0 _without_ that time axis dimension
    # and the loop will iterate over the years

    gpp_shape = gpp.shape
    if time_axis < 0 or time_axis >= len(gpp_shape):
        raise RuntimeError(f'time_axis must be >= 0 and <= {len(gpp_shape) - 1}')

    # Check that the input shapes for a single year match the shape of the
    # initial diameters
    single_year_gpp = np.take(gpp, indices=0, axis=time_axis)
    _ = check_input_shapes(single_year_gpp, d_init)

    # TODO - handle 1D GPP time series applied to more than one diameter

    # Initialise the Tree object
    tree = TTree(traits)

    # Check the requested outvars
    if 'diameter' not in outvars:
        raise RuntimeWarning('output_vars must include diameter')

    badvars = []
    for var in outvars:

        try:
            _ = getattr(tree, var)
        except AttributeError:
            badvars.append(var)

    if badvars:
        raise RuntimeError(f"Unknown tree properties in outvars: {', '.join(badvars)}")

    # Create an array to store the requested variables by adding a dimension
    # to the gpp input with length set to the number of variables
    output_shape = gpp_shape + tuple([len(outvars)])
    output = np.zeros(output_shape)

    # Insert the initial diameters
    tree.set_diameter(d_init)

    # Create an indexing object to insert values into the output. This is
    # a bit obscure: the inputs have a time axis and an arbitrary number
    # of other dimensions and the output adds another dimension for the
    # output variables. So this object creates a slice (:) on _all_ dimensions
    # and the loop then replaces the time axis and variable axis with integers.
    output_index = [slice(None)] * output.ndim

    # Loop over the gpp time axis
    for year in np.arange(gpp_shape[time_axis]):

        # Calculate the growth based on the current year of gpp
        tree.calculate_growth(np.take(gpp, indices=year, axis=time_axis))

        # Store the requested variables into the output array
        for var_idx, each_var in enumerate(outvars):

            # Extract variable values from tree into output - set the last index
            # (variable dimension) to the variable index and the time axis to the year
            output_index[-1] = var_idx
            output_index[time_axis] = year
            output[tuple(output_index)] = getattr(tree, each_var)

        # Now update the tree object
        tree.set_diameter(tree.diameter + getattr(tree, 'delta_d'))

    return output
