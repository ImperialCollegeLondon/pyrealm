"""The module :mod:`~pyrealm.pmodel.pmodel` provides the implementation of
the following pmodel core class:

* :class:`~pyrealm.pmodel.pmodel.PModel`:
    Applies the PModel to locations.
"""  # noqa D210, D415

from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst, PModelConst
from pyrealm.core.utilities import check_input_shapes, summarize_attrs
from pyrealm.pmodel.functions import (
    calc_ftemp_inst_rd,
    calc_ftemp_inst_vcmax,
    calc_ftemp_kphio,
)
from pyrealm.pmodel.jmax_limitation import JmaxLimitation
from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY, OptimalChiABC
from pyrealm.pmodel.pmodel_environment import PModelEnvironment

# Design notes on PModel (0.3.1 -> 0.4.0)
# The PModel until 0.3.1 was a single class taking tc etc. as inputs. However
# a common use case would be to look at how the PModel predictions change with
# different options. I (DO) did consider retaining the single class and having
# PModel __init__ create the environment and then have PModel.fit(), but
# having two classes seemed to better separate the physiological model (PModel
# class and attributes) from the environment the model is being fitted to and
# also creates _separate_ PModel objects.

# Design notes on PModel (0.4.0 -> 0.5.0)
# In separating PPFD and FAPAR into a second step, I had created the IabsScaled
# class to store variables that scale linearly with Iabs. That class held and
# exposed scaled and unscaled versions of several parameteres. For a start, that
# was a bad class name since the values could be unscaled, but also most of the
# unscaled versions are never really used. Really (hat tip, Keith Bloomfield),
# there are two meaningful _efficiency_ variables (LUE, IWUE) and then a set of
# productivity variables. The new structure reflects that, removing the
# un-needed unscaled variables and simplifying the structure.


class PModel:
    r"""Fit the P Model.

    This class fits the P Model to a given set of environmental and photosynthetic
    parameters. The calculated attributes of the class are described below. An extended
    description with typical use cases is given in :any:`pmodel_overview` but the basic
    flow of the model is:

    1. Estimate :math:`\ce{CO2}` limitation factors and optimal internal to ambient
       :math:`\ce{CO2}` partial pressure ratios (:math:`\chi`), using one of the
       methods based on :class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC`.
    2. Estimate limitation factors to :math:`V_{cmax}` and :math:`J_{max}` using
       :class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitation`.
    3. Optionally, estimate productivity measures including GPP by supplying FAPAR and
       PPFD using the :meth:`~pyrealm.pmodel.pmodel.PModel.estimate_productivity`
       method.

    The model predictions from step 1 and 2 are then:

    * Intrinsic water use efficiency (iWUE, :math:`\mu\mathrm{mol}\;\mathrm{mol}^{-1}`),
      calculated as :math:`( 5/8 * (c_a - c_i)) / P`, where `c_a` and `c_i` are measured
      in Pa and :math:`P` is atmospheric pressure in megapascals. This is equivalent to
      :math:`(c_a - c_i)/1.6` when `c_a` and `c_i` are expressed as parts per million.

    * The light use efficienciy (LUE, gC mol-1), calculated as:

        .. math::

            \text{LUE} = \phi_0 \cdot m_j \cdot f_v \cdot M_C

      where :math:`f_v` is a limitation factor defined in
      :class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitation` and :math:`M_C` is the
      molar mass
      of carbon.

    After running :meth:`~pyrealm.pmodel.pmodel.PModel.estimate_productivity`, the
    following predictions are also populated:

    * Gross primary productivity, calculated as :math:`\text{GPP} = \text{LUE} \cdot
      I_{abs}`, where :math:`I_{abs}` is the absorbed photosynthetic radiation

    * The maximum rate of Rubisco regeneration at the growth temperature
      (:math:`J_{max}`)

    * The maximum carboxylation capacity (mol C m-2) at the growth temperature
      (:math:`V_{cmax}`).

    These two predictions are calculated as follows:

        .. math::
            :nowrap:

            \[
                \begin{align*} V_{cmax} &= \phi_{0} I_{abs} \frac{m}{m_c} f_{v} \\
                J_{max} &= 4 \phi_{0} I_{abs} f_{j} \\ \end{align*}
            \]

    where  :math:`f_v, f_j` are limitation terms described in
    :class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitation`

    * The maximum carboxylation capacity (mol C m-2) normalised to the standard
      temperature as: :math:`V_{cmax25} = V_{cmax}  / fv(t)`, where :math:`fv(t)` is the
      instantaneous temperature response of :math:`V_{cmax}` implemented in
      :func:`~pyrealm.pmodel.functions.calc_ftemp_inst_vcmax`

    * Dark respiration, calculated as:

        .. math::

            R_d = b_0 \frac{fr(t)}{fv(t)} V_{cmax}

      following :cite:`Atkin:2015hk`, where :math:`fr(t)` is the instantaneous
      temperature response of dark respiration implemented in
      :func:`~pyrealm.pmodel.functions.calc_ftemp_inst_rd`, and :math:`b_0` is set in
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.atkin_rd_to_vcmax`.

    * Stomatal conductance (:math:`g_s`), calculated as:

        .. math::

            g_s = \frac{LUE}{M_C}\frac{1}{c_a - c_i}

      When C4 photosynthesis is being used, the true partial pressure of CO2 in the
      substomatal cavities (:math:`c_i`) is used following the calculation of
      :math:`\chi` using
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`. Note that
      :math:`g_s \to \infty` as VPD :math:`\to 0` and hence :math:`(c_a - c_i) \to 0`
      and the reported values will be set to ``np.nan`` under these conditions.

    Soil moisture effects:
        The `lavergne20_c3`, `lavergne20_c4`, ``prentice14_rootzonestress``,
        ``c4_rootzonestress`` and ``c4_no_gamma_rootzonestress`` options to
        ``method_optchi`` implement different approaches to soil moisture effects on
        photosynthesis. See also the alternative GPP penalty factors that can be applied
        after fitting the P Model
        (:func:`pyrealm.pmodel.functions.calc_soilmstress_stocker` and
        :func:`pyrealm.pmodel.functions.calc_soilmstress_mengoli`).

    Args:
        env: An instance of
         :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`
        kphio: (Optional) The quantum yield efficiency of photosynthesis
            (:math:`\phi_0`, unitless). Note that :math:`\phi_0` is sometimes used to
            refer to the quantum yield of electron transfer, which is exactly four times
            larger, so check definitions here.
        method_optchi: (Optional, default=`prentice14`) Selects the method to be
            used for calculating optimal :math:`chi`. The choice of method also sets the
            choice of  C3 or C4 photosynthetic pathway (see
            :class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC`).
        method_jmaxlim: (Optional, default=`wang17`) Method to use for
            :math:`J_{max}` limitation
        do_ftemp_kphio: (Optional, default=True) Include the temperature-
            dependence of quantum yield efficiency (see
            :func:`~pyrealm.pmodel.functions.calc_ftemp_kphio`).

    Examples:
        >>> import numpy as np
        >>> from pyrealm.pmodel.pmodel_environment import PModelEnvironment
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0])
        ... )
        >>> # Predictions from C3 P model
        >>> mod_c3 = PModel(env)
        >>> mod_c3.optchi.ci.round(5)
        array([28.14209])
        >>> mod_c3.optchi.chi.round(5)
        array([0.69435])
        >>> mod_c3.estimate_productivity(fapar=1, ppfd=300)
        >>> mod_c3.gpp.round(5)
        array([76.42545])
        >>> # Predictions from C4 P model
        >>> mod_c4 = PModel(env, method_optchi='c4', method_jmaxlim='none')
        >>> mod_c4.optchi.ci.round(5)
        array([18.22528])
        >>> mod_c4.optchi.chi.round(5)
        array([0.44967])
        >>> mod_c4.estimate_productivity(fapar=1, ppfd=300)
        >>> mod_c4.gpp.round(5)
        array([103.25886])
    """

    def __init__(
        self,
        env: PModelEnvironment,
        kphio: float | None = None,
        do_ftemp_kphio: bool = True,
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
    ):
        self.shape: tuple = env.shape
        """Records the common numpy array shape of array inputs."""

        # Store a reference to the photosynthetic environment and a direct
        # reference to the parameterisation
        self.env: PModelEnvironment = env
        """The PModelEnvironment used to fit the P Model."""

        self.pmodel_const: PModelConst = env.pmodel_const
        """The PModelConst instance used to create the model environment."""
        self.core_const: CoreConst = env.core_const
        """The CoreConst instance used to create the model environment."""

        # kphio calculation:
        self.init_kphio: float
        r"""The initial value of :math:`\phi_0` (``kphio``)"""
        self.kphio: NDArray
        r"""The value of :math:`\phi_0` used with any temperature correction applied."""
        self.do_ftemp_kphio: bool = do_ftemp_kphio
        r"""Records if :math:`\phi_0` (``kphio``) is temperature corrected."""

        # Set context specific defaults for kphio to match Stocker paper
        if kphio is None:
            if not self.do_ftemp_kphio:
                self.init_kphio = 0.049977
            else:
                self.init_kphio = 0.081785
        else:
            self.init_kphio = kphio

        # -----------------------------------------------------------------------
        # Optimal ci
        # The heart of the P-model: calculate ci:ca ratio (chi) and additional terms
        # -----------------------------------------------------------------------
        self.method_optchi: str = method_optchi
        """Records the method used to calculate optimal chi."""

        try:
            opt_chi_class = OPTIMAL_CHI_CLASS_REGISTRY[method_optchi]
        except KeyError:
            raise ValueError(f"Unknown optimal chi estimation method: {method_optchi}")

        self.optchi: OptimalChiABC = opt_chi_class(
            env=env,
            pmodel_const=self.pmodel_const,
        )
        """An subclass OptimalChi, implementing the requested chi calculation method"""

        self.c4: bool = self.optchi.is_c4
        """Does the OptimalChi method approximate a C3 or C4 pathway."""

        # -----------------------------------------------------------------------
        # Temperature dependence of quantum yield efficiency
        # -----------------------------------------------------------------------
        if self.do_ftemp_kphio:
            ftemp_kphio = calc_ftemp_kphio(
                env.tc, self.c4, pmodel_const=self.pmodel_const
            )
            self.kphio = self.init_kphio * ftemp_kphio
        else:
            self.kphio = np.array([self.init_kphio])

        # -----------------------------------------------------------------------
        # Calculation of Jmax limitation terms
        # -----------------------------------------------------------------------
        self.method_jmaxlim: str = method_jmaxlim
        """Records the method used to calculate Jmax limitation."""

        self.jmaxlim: JmaxLimitation = JmaxLimitation(
            self.optchi, method=self.method_jmaxlim, pmodel_const=self.pmodel_const
        )
        """Details of the Jmax limitation calculation for the model"""
        # -----------------------------------------------------------------------
        # Store the two efficiency predictions
        # -----------------------------------------------------------------------

        # Intrinsic water use efficiency (in µmol mol-1). The rpmodel reports this
        # in Pascals, but more commonly reported in µmol mol-1. The standard equation
        # (ca - ci) / 1.6 expects inputs in ppm, so the pascal versions are back
        # converted here.
        self.iwue: NDArray = (5 / 8 * (env.ca - self.optchi.ci)) / (
            1e-6 * self.env.patm
        )
        """Intrinsic water use efficiency (iWUE, µmol mol-1)"""

        # The basic calculation of LUE = phi0 * M_c * m with an added penalty term
        # for jmax limitation
        self.lue: NDArray = (
            self.kphio * self.optchi.mj * self.jmaxlim.f_v * self.core_const.k_c_molmass
        )
        """Light use efficiency (LUE, g C mol-1)"""

        # -----------------------------------------------------------------------
        # Define attributes populated by estimate_productivity method - these have
        # no defaults and are only populated by estimate_productivity. Their getter
        # methods have a check to raise an informative error
        # -----------------------------------------------------------------------
        self._vcmax: NDArray
        self._vcmax25: NDArray
        self._rd: NDArray
        self._jmax: NDArray
        self._gpp: NDArray
        self._gs: NDArray
        self._ppfd: NDArray
        self._fapar: NDArray

    def _check_estimated(self, varname: str) -> None:
        """Raise error when accessing unpopulated parameters.

        A helper function to raise an error when a user accesses a P Model
        parameter that has not yet been estimated via `estimate_productivity`.
        """
        if not hasattr(self, "_" + varname):
            raise RuntimeError(f"{varname} not calculated: use estimate_productivity")

    @property
    def gpp(self) -> NDArray:
        """Gross primary productivity (µg C m-2 s-1)."""
        self._check_estimated("gpp")

        return self._gpp

    @property
    def vcmax(self) -> NDArray:
        """Maximum rate of carboxylation (µmol m-2 s-1)."""
        self._check_estimated("vcmax")
        return self._vcmax

    @property
    def vcmax25(self) -> NDArray:
        """Maximum rate of carboxylation at standard temperature (µmol m-2 s-1)."""
        self._check_estimated("vcmax25")
        return self._vcmax25

    @property
    def rd(self) -> NDArray:
        """Dark respiration (µmol m-2 s-1)."""
        self._check_estimated("rd")
        return self._rd

    @property
    def jmax(self) -> NDArray:
        """Maximum rate of electron transport (µmol m-2 s-1)."""
        self._check_estimated("jmax")
        return self._jmax

    @property
    def gs(self) -> NDArray:
        """Stomatal conductance (µmol m-2 s-1)."""
        self._check_estimated("gs")
        return self._gs

    @property
    def ppfd(self) -> NDArray:
        """Photosynthetic photon flux density (PPFD, µmol m-2 s-1)."""
        self._check_estimated("gs")
        return self._ppfd

    @property
    def fapar(self) -> NDArray:
        """Fraction of absorbed photosynthetically active radiation
        (:math:`f_{APAR}` unitless).
        """  # noqa: D205
        self._check_estimated("gs")
        return self._fapar

    def estimate_productivity(
        self, fapar: np.ndarray = np.array([1]), ppfd: np.ndarray = np.array([1])
    ) -> None:
        r"""Estimate productivity of P Model from absorbed irradiance.

        This method takes the light use efficiency and Vcmax per unit absorbed
        irradiance and populates the following PModel attributes:
        :attr:`~pyrealm.pmodel.pmodel.PModel.gpp`,
        :attr:`~pyrealm.pmodel.pmodel.PModel.rd`,
        :attr:`~pyrealm.pmodel.pmodel.PModel.vcmax`,
        :attr:`~pyrealm.pmodel.pmodel.PModel.vcmax25`,
        :attr:`~pyrealm.pmodel.pmodel.PModel.jmax` and
        :attr:`~pyrealm.pmodel.pmodel.PModel.gs`.

        The function finds the total absorbed irradiance (:math:`I_{abs}`) as the
        product of the photosynthetic photon flux density (PPFD, `ppfd`) and the
        fraction of absorbed photosynthetically active radiation (`fapar`). The default
        values of ``ppfd`` and ``fapar`` provide estimates of the variables above per
        unit absorbed irradiance.

        PPFD _must_ be provided in units of micromols per metre square per second (µmol
        m-2 s-1). This is required to ensure that values of :math:`J_{max}` and
        :math:`V_{cmax}` are also in µmol m-2 s-1.

        Args:
            fapar: the fraction of absorbed photosynthetically active radiation (-)
            ppfd: photosynthetic photon flux density (µmol m-2 s-1)
        """

        # Check input shapes against each other and an existing calculated value
        _ = check_input_shapes(ppfd, fapar, self.lue)

        # Store the input ppfd and fapar - this is primarily so that they can be reused
        # by the subdaily model
        self._fapar = fapar
        self._ppfd = ppfd

        # Calculate Iabs
        iabs = fapar * ppfd

        # GPP
        self._gpp = self.lue * iabs

        # V_cmax
        self._vcmax = self.kphio * iabs * self.optchi.mjoc * self.jmaxlim.f_v

        # V_cmax25 (vcmax normalized to const.k_To)
        ftemp25_inst_vcmax = calc_ftemp_inst_vcmax(
            self.env.tc, core_const=self.core_const, pmodel_const=self.pmodel_const
        )
        self._vcmax25 = self._vcmax / ftemp25_inst_vcmax

        # Dark respiration at growth temperature
        ftemp_inst_rd = calc_ftemp_inst_rd(self.env.tc, pmodel_const=self.pmodel_const)
        self._rd = (
            self.pmodel_const.atkin_rd_to_vcmax
            * (ftemp_inst_rd / ftemp25_inst_vcmax)
            * self._vcmax
        )

        # Calculate Jmax
        self._jmax = 4 * self.kphio * iabs * self.jmaxlim.f_j

        # AJ and AC
        a_j = self.kphio * iabs * self.optchi.mj * self.jmaxlim.f_v
        a_c = self._vcmax * self.optchi.mc

        assim = np.minimum(a_j, a_c)

        if not np.allclose(
            assim, self._gpp / self.core_const.k_c_molmass, equal_nan=True
        ):
            warn("Assimilation and GPP are not identical")

        # Stomatal conductance - do not estimate when VPD = 0 or when floating point
        # errors give rise to (ca - ci) < 0 and deliberately ignore the numpy divide by
        # zero warnings in those cases.
        ca_ci_diff = self.env.ca - self.optchi.ci
        with np.errstate(divide="ignore", invalid="ignore"):
            self._gs = np.where(
                np.logical_and(self.env.vpd > 0, ca_ci_diff > 0),
                assim / ca_ci_diff,
                np.nan,
            )

    def __repr__(self) -> str:
        """Generates a string representation of PModel instance."""

        return (
            f"PModel("
            f"shape={self.shape}, "
            f"initial kphio={self.init_kphio}, "
            f"ftemp_kphio={self.do_ftemp_kphio}, "
            f"method_optchi={self.method_optchi}, "
            f"c4={self.c4}, "
            f"method_jmaxlim={self.method_jmaxlim}, "
        )

    def summarize(self, dp: int = 2) -> None:
        """Prints a summary of PModel estimates.

        Prints a summary of the calculated values in a PModel instance including the
        mean, range and number of nan values. This will always show efficiency variables
        (LUE and IWUE) and productivity estimates are shown if
        :meth:`~pyrealm.pmodel.pmodel.PModel.estimate_productivity` has been run.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [("lue", "g C mol-1"), ("iwue", "µmol mol-1")]

        if hasattr(self, "_gpp"):
            attrs.extend(
                [
                    ("gpp", "µg C m-2 s-1"),
                    ("vcmax", "µmol m-2 s-1"),
                    ("vcmax25", "µmol m-2 s-1"),
                    ("rd", "µmol m-2 s-1"),
                    ("gs", "µmol m-2 s-1"),
                    ("jmax", "µmol m-2 s-1"),
                ]
            )

        summarize_attrs(self, attrs, dp=dp)
