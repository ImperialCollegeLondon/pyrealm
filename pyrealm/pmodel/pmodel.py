"""The :mod:`~pyrealm.pmodel.pmodel` submodule provides the core implementation of the
following core classes:

* :class:`~pyrealm.pmodel.pmodel.PModelEnvironment`:
    Calculates the photosynthetic environment for locations.
* :class:`~pyrealm.pmodel.pmodel.PModel`:
    Applies the PModel to locations
* :class:`~pyrealm.pmodel.pmodel.CalcOptimalChi`:
    Estimates the optimal chi for locations, given an estimation method and settings
* :class:`~pyrealm.pmodel.pmodel.JmaxLimitation`:
    Estimates the Jmax limitation, given a method and settings
"""  # noqa D210, D415

from typing import Optional, Union
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm import ExperimentalFeatureWarning
from pyrealm.constants import PModelConst
from pyrealm.pmodel.functions import (
    calc_co2_to_ca,
    calc_ftemp_inst_rd,
    calc_ftemp_inst_vcmax,
    calc_ftemp_kphio,
    calc_gammastar,
    calc_kmm,
    calc_ns_star,
)
from pyrealm.utilities import bounds_checker, check_input_shapes, summarize_attrs

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


class PModelEnvironment:
    r"""Create a PModelEnvironment instance.

    This class takes the four key environmental inputs to the P Model and
    calculates four photosynthetic variables for those environmental
    conditions:

    * the photorespiratory CO2 compensation point (:math:`\Gamma^{*}`,
      using :func:`~pyrealm.pmodel.functions.calc_gammastar`),
    * the relative viscosity of water (:math:`\eta^*`,
      using :func:`~pyrealm.pmodel.functions.calc_ns_star`),
    * the ambient partial pressure of :math:`\ce{CO2}` (:math:`c_a`,
      using :func:`~pyrealm.pmodel.functions.calc_co2_to_ca`) and
    * the Michaelis Menten coefficient of Rubisco-limited assimilation
      (:math:`K`, using :func:`~pyrealm.pmodel.functions.calc_kmm`).

    These variables can then be used to fit P models using different
    configurations. Note that the underlying constants of the P Model
    (:class:`~pyrealm.constants.pmodel_const.PModelConst`) are set when creating
    an instance of this class.

    In addition to the four key variables above, the PModelEnvironment class
    is used to provide additional variables used by some methods.

    * the volumetric soil moisture content, required to calculate optimal
      :math:`\chi` in :meth:`~pyrealm.pmodel.pmodel.CalcOptimalChi.lavergne20_c3`.

    Args:
        tc: Temperature, relevant for photosynthesis (°C)
        vpd: Vapour pressure deficit (Pa)
        co2: Atmospheric CO2 concentration (ppm)
        patm: Atmospheric pressure (Pa)
        theta: Volumetric soil moisture (m3/m3)
        const: An instance of
            :class:`~pyrealm.constants.pmodel_const.PModelConst`.
    """

    def __init__(
        self,
        tc: NDArray,
        vpd: NDArray,
        co2: NDArray,
        patm: NDArray,
        theta: Optional[NDArray] = None,
        const: PModelConst = PModelConst(),
    ):
        self.shape: tuple = check_input_shapes(tc, vpd, co2, patm)

        # Validate and store the forcing variables
        self.tc: NDArray = bounds_checker(tc, -25, 80, "[]", "tc", "°C")
        """The temperature at which to estimate photosynthesis, °C"""
        self.vpd: NDArray = bounds_checker(vpd, 0, 10000, "[]", "vpd", "Pa")
        """Vapour pressure deficit, Pa"""
        self.co2: NDArray = bounds_checker(co2, 0, 1000, "[]", "co2", "ppm")
        """CO2 concentration, ppm"""
        self.patm: NDArray = bounds_checker(patm, 30000, 110000, "[]", "patm", "Pa")
        """Atmospheric pressure, Pa"""

        # Guard against calc_density issues
        if np.nanmin(self.tc) < -25:
            raise ValueError(
                "Cannot calculate P Model predictions for values below"
                " -25°C. See calc_density_h2o."
            )

        # Guard against negative VPD issues
        if np.nanmin(self.vpd) < 0:
            raise ValueError(
                "Negative VPD values will lead to missing data - clip to "
                "zero or explicitly set to np.nan"
            )

        self.ca: NDArray = calc_co2_to_ca(self.co2, self.patm)
        """Ambient CO2 partial pressure, Pa"""

        self.gammastar = calc_gammastar(tc, patm, const=const)
        r"""Photorespiratory compensation point (:math:`\Gamma^\ast`, Pa)"""

        self.kmm = calc_kmm(tc, patm, const=const)
        """Michaelis Menten coefficient, Pa"""

        # # Michaelis-Menten coef. C4 plants (Pa) NOT CHECKED. Need to think
        # # about how many optional variables stack up in PModelEnvironment
        # # and this is only required by C4 optimal chi Scott and Smith, which
        # # has not yet been implemented.
        # self.kp_c4 = calc_kp_c4(tc, patm, const=const)

        self.ns_star = calc_ns_star(tc, patm, const=const)
        """Viscosity correction factor realtive to standard
        temperature and pressure, unitless"""

        # Optional variables
        self.theta: Optional[NDArray]
        """Volumetric soil moisture (m3/m3)"""

        if theta is None:
            self.theta = None
        else:
            # Is the input congruent with the other variables and in bounds.
            _ = check_input_shapes(tc, theta)
            self.theta = bounds_checker(theta, 0, 0.8, "[]", "theta", "m3/m3")

        # Store parameters
        self.const = const
        """PModel Parameters used from calculation"""

    def __repr__(self) -> str:
        """Generates a string representation of PModelEnvironment instance."""
        # DESIGN NOTE: This is deliberately extremely terse. It could contain
        # a bunch of info on the environment but that would be quite spammy
        # on screen. Having a specific summary method that provides that info
        # is more user friendly.

        return f"PModelEnvironment(shape={self.shape})"

    def summarize(self, dp: int = 2) -> None:
        """Prints a summary of PModelEnvironment variables.

        Prints a summary of the input and photosynthetic attributes in a instance of a
        PModelEnvironment including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [
            ("tc", "°C"),
            ("vpd", "Pa"),
            ("co2", "ppm"),
            ("patm", "Pa"),
            ("ca", "Pa"),
            ("gammastar", "Pa"),
            ("kmm", "Pa"),
            ("ns_star", "-"),
        ]

        if self.theta is not None:
            attrs += [("theta", "m3/m3")]

        summarize_attrs(self, attrs, dp=dp)


class PModel:
    r"""Fit the P Model.

    This class fits the P Model to a given set of environmental and photosynthetic
    parameters. The calculated attributes of the class are described below. An extended
    description with typical use cases is given in :any:`pmodel_overview` but the basic
    flow of the model is:

    1. Estimate :math:`\ce{CO2}` limitation factors and optimal internal to ambient
       :math:`\ce{CO2}` partial pressure ratios (:math:`\chi`), using
       :class:`~pyrealm.pmodel.pmodel.CalcOptimalChi`.
    2. Estimate limitation factors to :math:`V_{cmax}` and :math:`J_{max}` using
       :class:`~pyrealm.pmodel.pmodel.JmaxLimitation`.
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

            \text{LUE} = \phi_0 \cdot m_j \cdot f_v \cdot M_C \cdot \beta(\theta),

      where :math:`f_v` is a limitation factor defined in
      :class:`~pyrealm.pmodel.pmodel.JmaxLimitation`, :math:`M_C` is the molar mass of
      carbon and :math:`\beta(\theta)` is an empirical soil moisture factor (see
      :func:`~pyrealm.pmodel.functions.calc_soilmstress`,  :cite:`Stocker:2020dh`).

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
    :class:`~pyrealm.pmodel.pmodel.JmaxLimitation`

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
        The `soilmstress`, `rootzonestress` arguments and the `lavergne20_c3` and
        `lavergne20_c4` all implement different approaches to soil moisture effects on
        photosynthesis and are incompatible.

    Args:
        env: An instance of :class:`~pyrealm.pmodel.pmodel.PModelEnvironment`.
        kphio: (Optional) The quantum yield efficiency of photosynthesis
            (:math:`\phi_0`, unitless). Note that :math:`\phi_0` is sometimes used to
            refer to the quantum yield of electron transfer, which is exactly four times
            larger, so check definitions here.
        rootzonestress: (Optional, default=None) An experimental option
            for providing a root zone water stress penalty to the :math:`beta` parameter
            in :class:`~pyrealm.pmodel.pmodel.CalcOptimalChi`.
        soilmstress: (Optional, default=None) A soil moisture stress factor
            calculated using :func:`~pyrealm.pmodel.functions.calc_soilmstress`.
        method_optchi: (Optional, default=`prentice14`) Selects the method to be
            used for calculating optimal :math:`chi`. The choice of method also sets the
            choice of  C3 or C4 photosynthetic pathway (see
            :class:`~pyrealm.pmodel.pmodel.CalcOptimalChi`).
        method_jmaxlim: (Optional, default=`wang17`) Method to use for
            :math:`J_{max}` limitation
        do_ftemp_kphio: (Optional, default=True) Include the temperature-
            dependence of quantum yield efficiency (see
            :func:`~pyrealm.pmodel.functions.calc_ftemp_kphio`).

    Examples:
        >>> env = PModelEnvironment(tc=20, vpd=1000, co2=400, patm=101325.0)
        >>> mod_c3 = PModel(env)
        >>> # Key variables from pmodel
        >>> np.round(mod_c3.optchi.ci, 5)
        28.14209
        >>> np.round(mod_c3.optchi.chi, 5)
        0.69435
        >>> mod_c3.estimate_productivity(fapar=1, ppfd=300)
        >>> np.round(mod_c3.gpp, 5)
        76.42545
        >>> mod_c4 = PModel(env, method_optchi='c4', method_jmaxlim='none')
        >>> # Key variables from PModel
        >>> np.round(mod_c4.optchi.ci, 5)
        18.22528
        >>> np.round(mod_c4.optchi.chi, 5)
        0.44967
        >>> mod_c4.estimate_productivity(fapar=1, ppfd=300)
        >>> np.round(mod_c4.gpp, 5)
        103.25886
    """

    def __init__(
        self,
        env: PModelEnvironment,
        rootzonestress: Optional[NDArray] = None,
        soilmstress: Optional[NDArray] = None,
        kphio: Optional[float] = None,
        do_ftemp_kphio: bool = True,
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
    ):
        # Check possible array inputs against the photosynthetic environment
        self.shape: tuple = check_input_shapes(
            env.gammastar, soilmstress, rootzonestress
        )
        """Records the common numpy array shape of array inputs."""

        # Store a reference to the photosynthetic environment and a direct
        # reference to the parameterisation
        self.env: PModelEnvironment = env
        """The PModelEnvironment used to fit the P Model."""

        self.const: PModelConst = env.const
        """The PModelConst instance used to create the model environment."""
        # ---------------------------------------------
        # Soil moisture and root zone stress handling
        # ---------------------------------------------

        if (
            (soilmstress is not None)
            + (rootzonestress is not None)
            + (method_optchi in ("lavergne20_c3", "lavergne20_c4"))
        ) > 1:
            raise AttributeError(
                "Soilmstress, rootzonestress and the lavergne20 method_optchi options "
                "are parallel approaches to soil moisture effects and cannot be "
                "combined."
            )

        if soilmstress is None:
            self.soilmstress: NDArray = np.array([1.0])
            """The soil moisture stress factor applied to model.

            This value will be 1.0 if no soil moisture stress was provided in the
            arguments to the class.
            """
            self._do_soilmstress: bool = False
            """Private flag indicating user provided soilmstress factor"""
        else:
            self.soilmstress = soilmstress
            self._do_soilmstress = True

        if rootzonestress is None:
            self._do_rootzonestress = False
            """Private flag indicating user provided rootzonestress factor"""
        else:
            warn(
                "The rootzonestress option is an experimental penalty factor to beta",
                ExperimentalFeatureWarning,
            )
            self.do_rootzonestress = True

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
            elif self._do_soilmstress:
                self.init_kphio = 0.087182
            else:
                self.init_kphio = 0.081785
        else:
            self.init_kphio = kphio

        # Check method_optchi and set c3/c4
        self.c4: bool = CalcOptimalChi._method_lookup(method_optchi)
        """Indicates if estimates calculated using C3 or C4 photosynthesis."""

        self.method_optchi: str = method_optchi
        """Records the method used to calculate optimal chi."""

        # -----------------------------------------------------------------------
        # Temperature dependence of quantum yield efficiency
        # -----------------------------------------------------------------------
        if self.do_ftemp_kphio:
            ftemp_kphio = calc_ftemp_kphio(env.tc, self.c4, const=env.const)
            self.kphio = self.init_kphio * ftemp_kphio
        else:
            self.kphio = np.array([self.init_kphio])

        # -----------------------------------------------------------------------
        # Optimal ci
        # The heart of the P-model: calculate ci:ca ratio (chi) and additional terms
        # -----------------------------------------------------------------------
        self.optchi: CalcOptimalChi = CalcOptimalChi(
            env=env,
            method=method_optchi,
            rootzonestress=rootzonestress or np.array([1.0]),
            const=env.const,
        )
        """Details of the optimal chi calculation for the model"""

        # -----------------------------------------------------------------------
        # Calculation of Jmax limitation terms
        # -----------------------------------------------------------------------
        self.method_jmaxlim: str = method_jmaxlim
        """Records the method used to calculate Jmax limitation."""

        self.jmaxlim: JmaxLimitation = JmaxLimitation(
            self.optchi, method=self.method_jmaxlim, const=env.const
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

        # The basic calculation of LUE = phi0 * M_c * m but here we implement
        # two penalty terms for jmax limitation and Stocker beta soil moisture
        # stress
        # Note: the rpmodel implementation also estimates soilmstress effects on
        #       jmax and vcmax but pyrealm.pmodel only applies the stress factor
        #       to LUE and hence GPP
        self.lue: NDArray = (
            self.kphio
            * self.optchi.mj
            * self.jmaxlim.f_v
            * self.const.k_c_molmass
            * self.soilmstress
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

    def _soilwarn(self, varname: str) -> None:
        """Emit warning about soil moisture stress factor.

        The empirical soil moisture stress factor (Stocker et al. 2020) _can_ be
        used to back calculate realistic Jmax and Vcmax values. The
        pyrealm.PModel implementation does not do so and this helper function is
        used to warn users within property getter functions
        """

        if self._do_soilmstress:
            warn(
                f"pyrealm.PModel does not correct {varname} for empirical soil "
                "moisture effects on LUE."
            )

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
        self._soilwarn("vcmax")
        return self._vcmax

    @property
    def vcmax25(self) -> NDArray:
        """Maximum rate of carboxylation at standard temperature (µmol m-2 s-1)."""
        self._check_estimated("vcmax25")
        self._soilwarn("vcmax25")
        return self._vcmax25

    @property
    def rd(self) -> NDArray:
        """Dark respiration (µmol m-2 s-1)."""
        self._check_estimated("rd")
        self._soilwarn("rd")
        return self._rd

    @property
    def jmax(self) -> NDArray:
        """Maximum rate of electron transport (µmol m-2 s-1)."""
        self._check_estimated("jmax")
        self._soilwarn("jmax")
        return self._jmax

    @property
    def gs(self) -> NDArray:
        """Stomatal conductance (µmol m-2 s-1)."""
        self._check_estimated("gs")
        self._soilwarn("gs")
        return self._gs

    def estimate_productivity(
        self, fapar: Union[float, np.ndarray] = 1, ppfd: Union[float, np.ndarray] = 1
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

        The functions finds the total absorbed irradiance (:math:`I_{abs}`) as the
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

        # Calculate Iabs
        iabs = fapar * ppfd

        # GPP
        self._gpp = self.lue * iabs

        # V_cmax
        self._vcmax = self.kphio * iabs * self.optchi.mjoc * self.jmaxlim.f_v

        # V_cmax25 (vcmax normalized to const.k_To)
        ftemp25_inst_vcmax = calc_ftemp_inst_vcmax(self.env.tc, const=self.const)
        self._vcmax25 = self._vcmax / ftemp25_inst_vcmax

        # Dark respiration at growth temperature
        ftemp_inst_rd = calc_ftemp_inst_rd(self.env.tc, const=self.const)
        self._rd = (
            self.const.atkin_rd_to_vcmax
            * (ftemp_inst_rd / ftemp25_inst_vcmax)
            * self._vcmax
        )

        # Calculate Jmax
        self._jmax = 4 * self.kphio * iabs * self.jmaxlim.f_j

        # AJ and AC
        a_j = self.kphio * iabs * self.optchi.mj * self.jmaxlim.f_v
        a_c = self._vcmax * self.optchi.mc

        assim = np.minimum(a_j, a_c)

        if not self._do_soilmstress and not np.allclose(
            assim, self._gpp / self.const.k_c_molmass, equal_nan=True
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
        if self._do_soilmstress:
            stress = "Soil moisture"
        elif self._do_rootzonestress:
            stress = "Root zone"
        else:
            stress = "None"
        return (
            f"PModel("
            f"shape={self.shape}, "
            f"initial kphio={self.init_kphio}, "
            f"ftemp_kphio={self.do_ftemp_kphio}, "
            f"method_optchi={self.method_optchi}, "
            f"c4={self.c4}, "
            f"method_jmaxlim={self.method_jmaxlim}, "
            f"Water stress={stress})"
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


class CalcOptimalChi:
    r"""Estimate optimal leaf internal CO2 concentration.

    This class provides alternative approaches to calculating the optimal :math:`\chi`
    and :math:`\ce{CO2}` limitation factors. These values are:

    - The optimal ratio of leaf internal to ambient :math:`\ce{CO2}` partial pressure
      (:math:`\chi = c_i/c_a`).
    - The :math:`\ce{CO2}` limitation term for light-limited assimilation (:math:`m_j`).
    - The :math:`\ce{CO2}` limitation term for Rubisco-limited assimilation
      (:math:`m_c`).

    The chosen method is automatically used to estimate these values when an instance is
    created - see the method documentation for individual details.

    The ratio of carboxylation to transpiration cost factors (``beta``, :math:`\beta`)
    is a key parameter in these methods. It is often held constant across cells but some
    methods (``lavergne20_c3`` and ``lavergne20_c4``) calculate :math:`\beta` from
    environmental conditions. For this reason, the ``beta`` attribute records the values
    used in calculations as an array.

    Args:
        env: An instance of PModelEnvironment providing the photosynthetic
          environment for the model.
        method: The method to use for estimating optimal :math:`\chi`, one
          of ``prentice14`` (default), ``lavergne20_c3``, ``c4``, ``c4_no_gamma`` or
          ``lavergne20_c4``.
        rootzonestress: This is an experimental feature to supply a root zone stress
          factor used as a direct penalty to :math:`\beta`, unitless. The default is
          1.0, with no root zone stress applied.
        const: An instance of
          :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    Returns:
        An instance of :class:`CalcOptimalChi` with the class attributes populated using
        the chosen methods and options.

    """

    def __init__(
        self,
        env: PModelEnvironment,
        rootzonestress: NDArray = np.array([1.0]),
        method: str = "prentice14",
        const: PModelConst = PModelConst(),
    ):
        self.env: PModelEnvironment = env
        """The PModelEnvironment containing the photosynthetic environment for the
        model."""

        # If rootzonestress is not simply equal to 1 (or an equivalent ndarray), check
        # rootzonestress conforms to the environment data
        if np.allclose(rootzonestress, 1.0):
            self.shape: tuple = env.shape
            """Records the common numpy array shape of array inputs."""
        else:
            self.shape = check_input_shapes(env.ca, rootzonestress)
            warn("The rootzonestress option is an experimental feature.")

        self.rootzonestress: NDArray = rootzonestress
        """Experimental rootzonestress factor, unitless."""

        # Declare attributes populated by methods - these attributes should never be
        # exposed without being populated as the method lookup below populates them
        # before leaving __init__, so they are not defined with a default value.
        self.beta: NDArray
        """The ratio of carboxylation to transpiration cost factors."""
        self.xi: NDArray
        r"""Defines the sensitivity of :math:`\chi` to the vapour pressure deficit,
        related to the carbon cost of water (Medlyn et al. 2011; Prentice et 2014)."""
        self.chi: NDArray
        r"""The ratio of leaf internal to ambient :math:`\ce{CO2}` partial pressure
        (:math:`\chi`)."""
        self.ci: NDArray
        r"""The leaf internal :math:`\ce{CO2}` partial pressure (:math:`c_i`)."""
        self.mc: NDArray
        r""":math:`\ce{CO2}` limitation factor for RuBisCO-limited assimilation
        (:math:`m_c`)."""
        self.mj: NDArray
        r""":math:`\ce{CO2}` limitation factor for light-limited assimilation
        (:math:`m_j`)."""
        self.mjoc: NDArray
        r"""Ratio of :math:`m_j/m_c`."""

        # TODO: considerable overlap between methods here - could maybe bring
        #       more into init but probably clearer and easier to debug to keep
        #       complete method code within methods.
        # TODO: Could convert this to use a registry?

        # Identify and run the selected method
        self.const: PModelConst = const
        """The PModelParams used for optimal chi estimation"""
        self.method: str = method
        """Records the method used for optimal chi estimation"""

        # Check the method - return value shows C3/C4 but not needed here - and
        # then run that method to populate the instance
        _ = self._method_lookup(method)
        method_func = getattr(self, method)
        method_func()

    @staticmethod
    def _method_lookup(method: str) -> bool:
        """Check a CalcOptimalChi method.

        This function validates a provided string, identifying a method of
        CalcOptimalChi, and also reports if the method is C4 or not.

        Args:
            method: A provided method name for
                :class:`~pyrealm.pmodel.pmodel.CalcOptimalChi`.

        Returns:
            A boolean showing if the method uses the C4 pathway.
        """

        map = {
            "prentice14": False,
            "lavergne20_c3": False,
            "c4": True,
            "c4_no_gamma": True,
            "lavergne20_c4": True,
        }

        is_c4 = map.get(method)

        if is_c4 is None:
            raise ValueError(f"CalcOptimalChi: method argument '{method}' invalid.")

        return is_c4

    def __repr__(self) -> str:
        """Generates a string representation of a CalcOptimalChi instance."""
        return f"CalcOptimalChi(shape={self.shape}, method={self.method})"

    def prentice14(self) -> None:
        r"""Calculate :math:`\chi` for C3 plants following :cite:`Prentice:2014bc`.

        Optimal :math:`\chi` is calculated following Equation 8 in
        :cite:`Prentice:2014bc`:

          .. math:: :nowrap:

            \[
                \begin{align*}
                    \chi &= \Gamma^{*} / c_a + (1- \Gamma^{*} / c_a)
                        \xi / (\xi + \sqrt D ), \text{where}\\
                    \xi &= \sqrt{(\beta (K+ \Gamma^{*}) / (1.6 \eta^{*}))}
                \end{align*}
            \]

        The :math:`\ce{CO2}` limitation term of light use efficiency (:math:`m_j`) is
        calculated following Equation 3 in :cite:`Wang:2017go`:

        .. math::

            m_j = \frac{c_a - \Gamma^{*}}
                       {c_a + 2 \Gamma^{*}}

        Finally,  :math:`m_c` is calculated, following Equation 7 in
        :cite:`Stocker:2020dh`, as:

        .. math::

            m_c = \frac{c_i - \Gamma^{*}}{c_i + K},

        where :math:`K` is the Michaelis Menten coefficient of Rubisco-limited
        assimilation.

        Examples:
            >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000)
            >>> vals = CalcOptimalChi(env=env)
            >>> round(vals.chi, 5)
            0.69435
            >>> round(vals.mc, 5)
            0.33408
            >>> round(vals.mj, 5)
            0.7123
            >>> round(vals.mjoc, 5)
            2.13211
        """

        # TODO: Docstring - equation for m_j previously included term now
        # removed from code:
        #     + 3 \Gamma^{*}
        #               \sqrt{\frac{1.6 D \eta^{*}}{\beta(K + \Gamma^{*})}}

        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.beta = self.const.beta_cost_ratio_prentice14
        self.xi = np.sqrt(
            (self.beta * self.rootzonestress * (self.env.kmm + self.env.gammastar))
            / (1.6 * self.env.ns_star)
        )

        self.chi = self.env.gammastar / self.env.ca + (
            1.0 - self.env.gammastar / self.env.ca
        ) * self.xi / (self.xi + np.sqrt(self.env.vpd))

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mj = (self.ci - self.env.gammastar) / (self.ci + 2 * self.env.gammastar)
        self.mc = (self.ci - self.env.gammastar) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc

    def lavergne20_c3(self) -> None:
        r"""Calculate soil moisture corrected :math:`\chi` for C3 plants.

        This method calculates the unit cost ratio $\beta$ as a function of soil
        moisture ($\theta$, m3 m-3), following :cite:`lavergne:2020a`:

          .. math:: :nowrap:

            \[
                \beta = e ^ {b\theta + a},
            \]

        The coefficients are experimentally derived values with defaults taken from
        Figure 6a of :cite:`lavergne:2020a` (:math:`a`,
        :attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_a_c3`;
        :math:`b`,
        :attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_b_c3`).

        Values of :math:`\chi` and other predictions are then calculated as in
        :meth:`~pyrealm.pmodel.pmodel.CalcOptimalChi.prentice14`. This method requires
        that `env` includes estimates of :math:`\theta` and  is incompatible with the
        `rootzonestress` approach.

        Examples:
            >>> env = PModelEnvironment(tc=20, patm=101325, co2=400,
            ...                         vpd=1000, theta=0.5)
            >>> vals = CalcOptimalChi(env=env, method='lavergne20_c3')
            >>> round(vals.beta, 5)
            224.75255
            >>> round(vals.chi, 5)
            0.73663
            >>> round(vals.mc, 5)
            0.34911
            >>> round(vals.mj, 5)
            0.7258
            >>> round(vals.mjoc, 5)
            2.07901
        """

        # This method needs theta
        if self.env.theta is None:
            raise RuntimeError(
                "Method `lavergne20_c3` requires soil moisture in the PModelEnvironment"
            )

        # Calculate beta as a function of theta
        self.beta = np.exp(
            self.const.lavergne_2020_b_c3 * self.env.theta
            + self.const.lavergne_2020_a_c3
        )

        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.xi = np.sqrt(
            (self.beta * (self.env.kmm + self.env.gammastar)) / (1.6 * self.env.ns_star)
        )

        self.chi = self.env.gammastar / self.env.ca + (
            1.0 - self.env.gammastar / self.env.ca
        ) * self.xi / (self.xi + np.sqrt(self.env.vpd))

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mj = (self.ci - self.env.gammastar) / (self.ci + 2 * self.env.gammastar)
        self.mc = (self.ci - self.env.gammastar) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc

    def lavergne20_c4(self) -> None:
        r"""Calculate soil moisture corrected :math:`\chi` for C4 plants.

        This method calculates :math:`\beta` as a function of soil moisture following
        the equation described in the
        :meth:`~pyrealm.pmodel.pmodel.CalcOptimalChi.lavergne20_c3` method.  However,
        the default coefficients of the moisture scaling from :cite:`lavergne:2020a` for
        C3 plants are adjusted to match the theoretical expectation that :math:`\beta`
        for C4 plants is nine times smaller than :math:`\beta` for C3 plants (see
        :meth:`~pyrealm.pmodel.pmodel.CalcOptimalChi.c4`): :math:`b`
        (:attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_b_c4`) is
        unchanged but
        :math:`a_{C4} = a_{C3} - log(9)`
        (:attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_a_c4`) .

        Following the calculation of :math:`\beta`, this method then follows the
        calculations described in
        :meth:`~pyrealm.pmodel.pmodel.CalcOptimalChi.c4_no_gamma`: :math:`m_j = 1.0`
        because photorespiration is negligible, but :math:`m_c` and hence
        :math:`m_{joc}` are calculated.

        Note:

        This is an **experimental approach**. The research underlying
        :cite:`lavergne:2020a`, found **no relationship** between C4 :math:`\beta`
        values and soil moisture in leaf gas exchange measurements.

        Examples:
            >>> env = PModelEnvironment(tc=20, patm=101325, co2=400,
            ...                         vpd=1000, theta=0.5)
            >>> vals = CalcOptimalChi(env=env, method='lavergne20_c4')
            >>> round(vals.beta, 5)
            24.97251
            >>> round(vals.chi, 5)
            0.44432
            >>> round(vals.mc, 5)
            0.28091
            >>> round(vals.mj, 5)
            1.0
            >>> round(vals.mjoc, 5)
            3.55989
        """

        # Warn that this is experimental
        warn(
            "The lavergne20_c4 method is experimental, see the method documentation",
            ExperimentalFeatureWarning,
        )

        # This method needs theta
        if self.env.theta is None:
            raise RuntimeError(
                "Method `lavergne20_c4` requires soil moisture in the PModelEnvironment"
            )

        # Calculate beta as a function of theta
        self.beta = np.exp(
            self.const.lavergne_2020_b_c4 * self.env.theta
            + self.const.lavergne_2020_a_c4
        )

        # Calculate chi and xi as in Prentice 14 but removing gamma terms.
        self.xi = np.sqrt((self.beta * self.env.kmm) / (1.6 * self.env.ns_star))
        self.chi = self.xi / (self.xi + np.sqrt(self.env.vpd))

        # mj is equal to 1 as gammastar is null
        self.mj = np.ones(self.shape)

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mc = (self.ci) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc

    def c4(self) -> None:
        r"""Estimate :math:`\chi` for C4 plants following :cite:`Prentice:2014bc`.

        Optimal :math:`\chi` is calculated as in
        :meth:`~pyrealm.pmodel.pmodel.CalcOptimalChi.prentice14`, but using a C4
        specific estimate of the unit cost ratio :math:`\beta`, see
        :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`.

        This method  sets :math:`m_j = m_c = m_{joc} = 1.0` to capture the
        boosted :math:`\ce{CO2}` concentrations at the chloropolast in C4
        photosynthesis.

        Examples:
            >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000)
            >>> vals = CalcOptimalChi(env=env, method='c4')
            >>> round(vals.chi, 5)
            0.44967
            >>> round(vals.mj, 1)
            1.0
            >>> round(vals.mc, 1)
            1.0
        """

        # Replace missing rootzonestress with 1
        self.rootzonestress = self.rootzonestress or np.array([1.0])

        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.beta = self.const.beta_cost_ratio_c4
        self.xi = np.sqrt(
            (self.beta * self.rootzonestress * (self.env.kmm + self.env.gammastar))
            / (1.6 * self.env.ns_star)
        )

        self.chi = self.env.gammastar / self.env.ca + (
            1.0 - self.env.gammastar / self.env.ca
        ) * self.xi / (self.xi + np.sqrt(self.env.vpd))

        self.ci = self.chi * self.env.ca

        # These values need to retain any dimensions of the original inputs - if
        # ftemp_kphio is set to 1.0 (i.e. no temperature correction) then the dimensions
        # of tc are lost.
        self.mc = np.ones(self.shape)
        self.mj = np.ones(self.shape)
        self.mjoc = np.ones(self.shape)

    def c4_no_gamma(self) -> None:
        r"""Calculate optimal chi assuming negligible photorespiration.

        This method assumes that photorespiration (:math:`\Gamma^\ast`) is negligible
        for C4 plants. This simplifies the calculation of :math:`\xi` and :math:`\chi`
        compared to :meth:`~pyrealm.pmodel.pmodel.CalcOptimalChi.c4`, but uses the same
        C4 specific estimate of the unit cost ratio :math:`\beta`,
        :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`.

          .. math:: :nowrap:

            \[
                \begin{align*}
                    \chi &= \xi / (\xi + \sqrt D ), \text{where}\\ \xi &= \sqrt{(\beta
                    K) / (1.6 \eta^{*}))}
                \end{align*}
            \]

        In addition, :math:`m_j = 1.0`  because photorespiration is negligible in C4
        photosynthesis, but :math:`m_c` and hence :math:`m_{joc}` are calculated, not
        set to one.

          .. math:: :nowrap:

            \[
                \begin{align*}
                    m_c &= \dfrac{\chi}{\chi + \frac{K}{c_a}}
                \end{align*}
            \]

        Examples:
            >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000)
            >>> vals = CalcOptimalChi(env=env, method='c4_no_gamma')
            >>> round(vals.chi, 5)
            0.3919
            >>> round(vals.mj, 1)
            1.0
            >>> round(vals.mc, 1)
            0.3
        """

        # Replace missing rootzonestress with 1
        self.rootzonestress = self.rootzonestress or np.array([1.0])

        # Calculate chi and xi as in Prentice 14 but removing gamma terms.
        self.beta = self.const.beta_cost_ratio_c4
        self.xi = np.sqrt(
            (self.beta * self.rootzonestress * self.env.kmm) / (1.6 * self.env.ns_star)
        )

        self.chi = self.xi / (self.xi + np.sqrt(self.env.vpd))

        # version 3 as in Scott & Smith (2022)
        # beta_c4 = 166
        # self.xi = np.sqrt((beta_c4 *self.env.kp_c4) / (1.6 * self.env.ns_star))
        # self.chi = self.xi /(self.xi + np.sqrt(self.env.vpd))

        # mj is equal to 1 as gammastar is null
        self.mj = np.ones(self.shape)

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mc = (self.ci) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc

    def summarize(self, dp: int = 2) -> None:
        """Print CalcOptimalChi summary.

        Prints a summary of the variables calculated within an instance
        of CalcOptimalChi including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [
            ("xi", " Pa ^ (1/2)"),
            ("chi", "-"),
            ("mc", "-"),
            ("mj", "-"),
            ("mjoc", "-"),
        ]
        summarize_attrs(self, attrs, dp=dp)


class JmaxLimitation:
    r"""Estimate Jmax limitation.

    This class calculates two factors (:math:`f_v` and :math:`f_j`) used to implement
    :math:`V_{cmax}` and :math:`J_{max}` limitation of photosynthesis. Three methods are
    currently implemented:

        * ``simple``: applies the 'simple' equations with no limitation. The alias
          ``none`` is also accepted.
        * ``wang17``: applies the framework of :cite:`Wang:2017go`.
        * ``smith19``: applies the framework of :cite:`Smith:2019dv`

    Note that :cite:`Smith:2019dv` defines :math:`\phi_0` as the quantum efficiency of
    electron transfer, whereas :class:`pyrealm.pmodel.pmodel.PModel` defines
    :math:`\phi_0` as the quantum efficiency of photosynthesis, which is 4 times
    smaller. This is why the factors here are a factor of 4 greater than Eqn 15 and 17
    in :cite:`Smith:2019dv`.

    Arguments:
        optchi: an instance of :class:`CalcOptimalChi` providing the :math:`\ce{CO2}`
            limitation term of light use efficiency (:math:`m_j`) and the
            :math:`\ce{CO2}` limitation term for Rubisco assimilation (:math:`m_c`).
        method: method to apply :math:`J_{max}` limitation (default: ``wang17``,
            or ``smith19`` or ``none``)
        const: An instance of :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    Examples:
        >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000)
        >>> optchi = CalcOptimalChi(env)
        >>> simple = JmaxLimitation(optchi, method='simple')
        >>> simple.f_j
        1.0
        >>> simple.f_v
        1.0
        >>> wang17 = JmaxLimitation(optchi, method='wang17')
        >>> round(wang17.f_j, 5)
        0.66722
        >>> round(wang17.f_v, 5)
        0.55502
        >>> smith19 = JmaxLimitation(optchi, method='smith19')
        >>> round(smith19.f_j, 5)
        1.10204
        >>> round(smith19.f_v, 5)
        0.75442
    """

    # TODO - apparent incorrectness of wang and smith methods with _ca_ variation,
    #        work well with varying temperature but not _ca_ variation (or
    #        e.g. elevation gradient David Sandoval, REALM meeting, Dec 2020)

    def __init__(
        self,
        optchi: CalcOptimalChi,
        method: str = "wang17",
        const: PModelConst = PModelConst(),
    ):
        self.shape: tuple = check_input_shapes(optchi.mj)
        """Records the common numpy array shape of array inputs."""
        self.optchi: CalcOptimalChi = optchi
        """Details of the optimal chi calculation for the model"""
        self.method: str = method
        """Records the method used to calculate Jmax limitation."""
        self.const: PModelConst = const
        """The PModelParams instance used for the calculation."""

        # Attributes populated by alternative method - two should always be populated by
        # the methods used below, but omega and omega_star only apply to smith19
        self.f_j: NDArray
        """:math:`J_{max}` limitation factor, calculated using the method."""
        self.f_v: NDArray
        """:math:`V_{cmax}` limitation factor, calculated using the method."""
        self.omega: Optional[NDArray] = None
        """Component of :math:`J_{max}` calculation for method ``smith19``
        (:cite:`Smith:2019dv`)."""
        self.omega_star: Optional[NDArray] = None
        """Component of :math:`J_{max}` calculation for method ``smith19``
        (:cite:`Smith:2019dv`)."""

        all_methods = {
            "wang17": self.wang17,
            "smith19": self.smith19,
            "simple": self.simple,
            "none": self.simple,
        }

        # Catch method errors.
        if self.method == "c4":
            raise ValueError(
                "This class does not implement a fixed method for C4 photosynthesis."
                "To replicate rpmodel choose method_optchi='c4' and method='simple'"
            )
        elif self.method not in all_methods:
            raise ValueError(f"JmaxLimitation: method argument '{method}' invalid.")

        # Use the selected method to calculate limitation factors
        this_method = all_methods[self.method]
        this_method()

    def __repr__(self) -> str:
        """Generates a string representation of a JmaxLimitation instance."""
        return f"JmaxLimitation(shape={self.shape})"

    def wang17(self) -> None:
        r"""Calculate limitation factors following :cite:`Wang:2017go`.

        These factors are described in Equation 49 of :cite:`Wang:2017go` as the
        square root term at the end of that equation:

            .. math::
                :nowrap:

                \[
                    \begin{align*}
                    f_v &=  \sqrt{ 1 - \frac{c^*}{m} ^{2/3}} \\
                    f_j &=  \sqrt{\frac{m}{c^*} ^{2/3} -1 } \\
                    \end{align*}
                \]

        The variable :math:`c^*` is a cost parameter for maintaining :math:`J_{max}`
        and is set in `const.wang_c`.
        """

        # Calculate √ {1 – (c*/m)^(2/3)} (see Eqn 2 of Wang et al 2017) and
        # √ {(m/c*)^(2/3) - 1} safely, both are undefined where m <= c*.
        vals_defined = np.greater(self.optchi.mj, self.const.wang17_c)

        self.f_v = np.sqrt(
            1 - (self.const.wang17_c / self.optchi.mj) ** (2.0 / 3.0),
            where=vals_defined,
        )
        self.f_j = np.sqrt(
            (self.optchi.mj / self.const.wang17_c) ** (2.0 / 3.0) - 1,
            where=vals_defined,
        )

        # Backfill undefined values - tackling float vs np.ndarray types
        if isinstance(self.f_v, np.ndarray):
            self.f_j[np.logical_not(vals_defined)] = np.nan  # type: ignore
            self.f_v[np.logical_not(vals_defined)] = np.nan  # type: ignore
        elif not vals_defined:
            self.f_j = np.nan
            self.f_v = np.nan

    def smith19(self) -> None:
        r"""Calculate limitation factors following :cite:`Smith:2019dv`.

        The values are calculated as:

        .. math::
            :nowrap:

            \[
                \begin{align*}
                f_v &=  \frac{\omega^*}{2\theta} \\
                f_j &=  \omega\\
                \end{align*}
            \]

        where,

        .. math::
            :nowrap:

            \[
                \begin{align*}
                \omega &= (1 - 2\theta) + \sqrt{(1-\theta)
                    \left(\frac{1}{\frac{4c}{m}(1 -
                    \theta\frac{4c}{m})}-4\theta\right)}\\
                \omega^* &= 1 + \omega - \sqrt{(1 + \omega) ^2 -4\theta\omega}
                \end{align*}
            \]

        given,

        * :math:`\theta`, (``const.smith19_theta``) captures the
          curved relationship between light intensity and photosynthetic
          capacity, and
        * :math:`c`, (``const.smith19_c_cost``) as a cost parameter
          for maintaining :math:`J_{max}`, equivalent to :math:`c^\ast = 4c`
          in the :meth:`~pyrealm.pmodel.pmodel.JmaxLimitation.wang17` method.
        """

        # Adopted from Nick Smith's code:
        # Calculate omega, see Smith et al., 2019 Ecology Letters  # Eq. S4
        theta = self.const.smith19_theta
        c_cost = self.const.smith19_c_cost

        # simplification terms for omega calculation
        cm = 4 * c_cost / self.optchi.mj
        v = 1 / (cm * (1 - self.const.smith19_theta * cm)) - 4 * theta

        # account for non-linearities at low m values. This code finds
        # the roots of a quadratic function that is defined purely from
        # the scalar theta, so will always be a scalar. The first root
        # is then used to set a filter for calculating omega.

        cap_p = (((1 / 1.4) - 0.7) ** 2 / (1 - theta)) + 3.4
        aquad = -1
        bquad = cap_p
        cquad = -(cap_p * theta)
        roots = np.polynomial.polynomial.polyroots(
            [aquad, bquad, cquad]
        )  # type: ignore [no-untyped-call]

        # factors derived as in Smith et al., 2019
        m_star = (4 * c_cost) / roots[0].real
        omega = np.where(
            self.optchi.mj < m_star,
            -(1 - (2 * theta)) - np.sqrt((1 - theta) * v),
            -(1 - (2 * theta)) + np.sqrt((1 - theta) * v),
        )

        # np.where _always_ returns an array, so catch scalars
        self.omega = omega.item() if np.ndim(omega) == 0 else omega

        self.omega_star = (
            1.0
            + self.omega
            - np.sqrt((1.0 + self.omega) ** 2 - (4.0 * theta * self.omega))  # Eq. 18
        )

        # Effect of Jmax limitation - note scaling here. Smith et al use
        # phi0 as as the quantum efficiency of electron transport, which is
        # 4 times our definition of phio0 as the quantum efficiency of photosynthesis.
        # So omega*/8 theta and omega / 4 are scaled down here  by a factor of 4.
        # Ignore `mypy` here as omega_star is explicitly not None.
        self.f_v = self.omega_star / (2.0 * theta)  # type: ignore
        self.f_j = self.omega

    def simple(self) -> None:
        """Apply the 'simple' form of the equations.

        This method allows the 'simple' form of the equations to be calculated
        by setting :math:`f_v = f_j = 1`.
        """

        # Set Jmax limitation to unity - could define as 1.0 in __init__ and
        # pass here, but setting explicitly within the method for clarity.
        self.f_v = np.array([1.0])
        self.f_j = np.array([1.0])
