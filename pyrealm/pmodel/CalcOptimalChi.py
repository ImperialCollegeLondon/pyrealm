"""The module :mod:`~pyrealm.pmodel.CalcOptimalChi` provides the implementation of 
the following pmodel core class:

* :class:`~pyrealm.pmodel.CalcOptimalChi.CalcOptimalChi`:
    Estimates the optimal chi for locations, given an estimation method and settings
"""  # noqa D210, D415

from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm import ExperimentalFeatureWarning
from pyrealm.constants import PModelConst
from pyrealm.pmodel.PModelEnvironment import PModelEnvironment
from pyrealm.utilities import check_input_shapes, summarize_attrs


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
                :class:`~pyrealm.pmodel.CalcOptimalChi.CalcOptimalChi`.

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
            >>> import numpy as np
            >>> env = PModelEnvironment(
            ...     tc=np.array([20]), vpd=np.array([1000]),
            ...     co2=np.array([400]), patm=np.array([101325.0])
            ... )
            >>> vals = CalcOptimalChi(env=env)
            >>> vals.chi.round(5)
            array([0.69435])
            >>> vals.mc.round(5)
            array([0.33408])
            >>> vals.mj.round(5)
            array([0.7123])
            >>> vals.mjoc.round(5)
            array([2.13211])
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
        :meth:`~pyrealm.pmodel.CalcOptimalChi.CalcOptimalChi.prentice14`. This method
        requires that `env` includes estimates of :math:`\theta` and  is incompatible
        with the `rootzonestress` approach.

        Examples:
            >>> import numpy as np
            >>> env = PModelEnvironment(
            ...     tc=np.array([20]), vpd=np.array([1000]), co2=np.array([400]),
            ...     patm=np.array([101325.0]), theta=np.array([0.5])
            ... )
            >>> vals = CalcOptimalChi(env=env, method='lavergne20_c3')
            >>> vals.beta.round(5)
            array([224.75255])
            >>> vals.chi.round(5)
            array([0.73663])
            >>> vals.mc.round(5)
            array([0.34911])
            >>> vals.mj.round(5)
            array([0.7258])
            >>> vals.mjoc.round(5)
            array([2.07901])
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
        :meth:`~pyrealm.pmodel.CalcOptimalChi.CalcOptimalChi.lavergne20_c3` method.
        However,
        the default coefficients of the moisture scaling from :cite:`lavergne:2020a` for
        C3 plants are adjusted to match the theoretical expectation that :math:`\beta`
        for C4 plants is nine times smaller than :math:`\beta` for C3 plants (see
        :meth:`~pyrealm.pmodel.CalcOptimalChi.CalcOptimalChi.c4`): :math:`b`
        (:attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_b_c4`) is
        unchanged but
        :math:`a_{C4} = a_{C3} - log(9)`
        (:attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_a_c4`) .

        Following the calculation of :math:`\beta`, this method then follows the
        calculations described in
        :meth:`~pyrealm.pmodel.CalcOptimalChi.CalcOptimalChi.c4_no_gamma`
        ::math:`m_j = 1.0`
        because photorespiration is negligible, but :math:`m_c` and hence
        :math:`m_{joc}` are calculated.

        Note:

        This is an **experimental approach**. The research underlying
        :cite:`lavergne:2020a`, found **no relationship** between C4 :math:`\beta`
        values and soil moisture in leaf gas exchange measurements.

        Examples:
            >>> import numpy as np
            >>> env = PModelEnvironment(
            ...     tc=np.array([20]), vpd=np.array([1000]), co2=np.array([400]),
            ...     patm=np.array([101325.0]), theta=np.array([0.5])
            ... )
            >>> vals = CalcOptimalChi(env=env, method='lavergne20_c4')
            >>> vals.beta.round(5)
            array([24.97251])
            >>> vals.chi.round(5)
            array([0.44432])
            >>> vals.mc.round(5)
            array([0.28091])
            >>> vals.mj.round(5)
            array([1.])
            >>> vals.mjoc.round(5)
            array([3.55989])
        """

        # Warn that this is experimental
        warn(
            "The lavergne20_c4 method is experimental, see the method " "documentation",
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
        :meth:`~pyrealm.pmodel.CalcOptimalChi.CalcOptimalChi.prentice14`, but using a C4
        specific estimate of the unit cost ratio :math:`\beta`, see
        :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`.

        This method  sets :math:`m_j = m_c = m_{joc} = 1.0` to capture the
        boosted :math:`\ce{CO2}` concentrations at the chloropolast in C4
        photosynthesis.

        Examples:
            >>> import numpy as np
            >>> env = PModelEnvironment(
            ...     tc=np.array([20]), vpd=np.array([1000]), co2=np.array([400]),
            ...     patm=np.array([101325.0]), theta=np.array([0.5])
            ... )
            >>> vals = CalcOptimalChi(env=env, method='c4')
            >>> vals.chi.round(5)
            array([0.44967])
            >>> vals.mj.round(1)
            array([1.])
            >>> vals.mc.round(1)
            array([1.])
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
        compared to :meth:`~pyrealm.pmodel.CalcOptimalChi.CalcOptimalChi.c4`, but uses
        the same
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
            >>> import numpy as np
            >>> env = PModelEnvironment(
            ...     tc=np.array([20]), vpd=np.array([1000]),
            ...     co2=np.array([400]), patm=np.array([101325.0])
            ... )
            >>> vals = CalcOptimalChi(env=env, method='c4_no_gamma')
            >>> vals.chi.round(5)
            array([0.3919])
            >>> vals.mj.round(1)
            array([1.])
            >>> vals.mc.round(1)
            array([0.3])
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
