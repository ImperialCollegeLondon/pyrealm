r"""The module :mod:`~pyrealm.pmodel.optimal_chi` provides
the abstract base class :class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC`,
which is used to support different implementations of the calculation of optimal chi.
"""  # noqa D210, D415

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import PModelConst
from pyrealm.core.experimental import warn_experimental
from pyrealm.core.utilities import check_input_shapes, summarize_attrs
from pyrealm.pmodel.pmodel_environment import PModelEnvironment

OPTIMAL_CHI_CLASS_REGISTRY: dict[str, type[OptimalChiABC]] = {}
"""A registry for optimal chi calculation classes.

Different implementations of the calculation of optimal chi must all be subclasses of
:class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC` abstract base class.
This dictionary is used as a registry for defined subclasses and a method name
is used to retrieve a particular implementation from this registry. For example:

.. code:: python

    prentice14_opt_chi = OPTIMAL_CHI_CLASS_REGISTRY['prentice14']
"""


class OptimalChiABC(ABC):
    r"""ABC for estimating optimal leaf internal :math:`\ce{CO2}` concentration.

    This provides an abstract base class for the implementation of alternative
    approaches to calculating the optimal :math:`\chi` and :math:`\ce{CO2}` limitation
    factors. All implementations estimate the following values, which are attributes of
    the resulting class instance.

    - The ratio of carboxylation to transpiration cost factors (``beta``,
      :math:`\beta`).
    - The variable ``xi`` (:math:`\xi`), which captures the sensitivity of :math:`\chi`
      to the vapour pressure deficit, and is related to the carbon cost of water (Medlyn
      et al. 2011; Prentice et 2014).
    - The optimal ratio of :math:`\ce{CO2}` partial pressures within the leaf (``ci``,
      :math:`c_i`) to the external environmental partial pressure (``ca```, :math:`c_i`)
      recorded. The optimal ratio ``chi`` (:math:`\chi = c_i/c_a`) is stored as well as
      the resulting internal partial pressure (``ci``).
    - The :math:`\ce{CO2}` limitation term for light-limited assimilation (``mj``,
      :math:`m_j`), the :math:`\ce{CO2}` limitation term for Rubisco-limited
      assimilation (``mc``, :math:`m_c`) and their ratio (``mjoc``, :math:`m_j/m_c`).

    The abstract base class requires that implementations of specific approaches define
    two methods:

    - `set_beta`: This method defines the calculation of the ``beta`` values to be used.
      The method is called by the ``__init__`` method when a subclass instance is
      created and should not change.
    - `estimate_chi`: This method defines the calculation of ``xi`` and then ``chi``,
      ``ci``, ``mj``, ``mc`` and ``mjoc``. This method is also called by the
      ``__init__`` method when a subclass instance is created, using the default defined
      calculation of ``xi``. However, it can also be called using modified values for
      ``xi`` to allow recalculation of these values. This is used primarily in fitting
      the P Model at subdaily scales, where the ``xi`` parameter acclimates slowly to
      changing environmental conditions.

    Args:
        env: An instance of
            :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`  providing the
            photosynthetic environment for the model.
        pmodel_const: An instance of
            :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    Returns:
        Instances of the abstract base class should not be created - use instances of
        specific subclasses.
    """

    method: str
    """A short method name used to identify the class in
    :data:`~pyrealm.pmodel.optimal_chi.OPTIMAL_CHI_CLASS_REGISTRY`.
    """
    is_c4: bool
    """A flag indicating if the method captures the C4 photosynthetic pathway."""
    requires: list[str]
    """A list of names of optional attributes of
    :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` that must be populated
    to use a method.
    """

    def __init__(
        self,
        env: PModelEnvironment,
        pmodel_const: PModelConst = PModelConst(),
    ):
        self.env: PModelEnvironment = env
        """The PModelEnvironment containing the photosynthetic environment for the
        model."""
        self.shape: tuple[int, ...] = env.shape
        """The shape of the input environment data."""

        self.pmodel_const: PModelConst = pmodel_const
        """The PModelParams used for optimal chi estimation"""

        # Declare attributes populated by methods. These are typed but not assigned a
        # default value as they must be populated by the set_beta and estimate_chi
        # methods, which are called below, and so will be populated before __init__
        # returns.
        self.beta: NDArray[np.float64]
        """The ratio of carboxylation to transpiration cost factors."""
        self.xi: NDArray[np.float64]
        r"""Defines the sensitivity of :math:`\chi` to the vapour pressure deficit,
        related to the carbon cost of water (Medlyn et al. 2011; Prentice et 2014)."""
        self.chi: NDArray[np.float64]
        r"""The ratio of leaf internal to ambient :math:`\ce{CO2}` partial pressure
        (:math:`\chi`)."""
        self.mc: NDArray[np.float64]
        r""":math:`\ce{CO2}` limitation factor for RuBisCO-limited assimilation
        (:math:`m_c`)."""
        self.mj: NDArray[np.float64]
        r""":math:`\ce{CO2}` limitation factor for light-limited assimilation
        (:math:`m_j`)."""
        self.ci: NDArray[np.float64]
        r"""The leaf internal :math:`\ce{CO2}` partial pressure (:math:`c_i`)."""
        self.mjoc: NDArray[np.float64]
        r"""Ratio of :math:`m_j/m_c`."""

        # Run the calculation methods after checking for any required variables
        self._check_requires()
        self.set_beta()
        self.estimate_chi()

        # Validate that the subclass methods populate the attributes correctly.
        _ = check_input_shapes(
            env.ca, self.beta, self.xi, self.chi, self.ci, self.mj, self.mc, self.mjoc
        )

    @abstractmethod
    def set_beta(self) -> None:
        """Set the beta values."""

    @abstractmethod
    def estimate_chi(self, xi_values: NDArray[np.float64] | None = None) -> None:
        """Estimate xi, chi and other variables."""

    def _check_requires(self) -> None:
        """Check additional required variables are present."""

        for required_var in self.requires:
            if not hasattr(self.env, required_var):
                raise ValueError(
                    f"{self.__class__.__name__} (method {self.method}) requires "
                    f"{required_var} to be provided in the PModelEnvironment."
                )

    def __repr__(self) -> str:
        """Generates a string representation of an OptimalChi instance."""
        return f"{type(self).__name__}(shape={self.shape})"

    def summarize(self, dp: int = 2) -> None:
        """Print OptimalChi summary.

        Prints a summary of the variables calculated within an instance
        of OptimalChi including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = (
            ("xi", " Pa ^ (1/2)"),
            ("chi", "-"),
            ("mc", "-"),
            ("mj", "-"),
            ("mjoc", "-"),
        )
        summarize_attrs(self, attrs, dp=dp)

    @classmethod
    def __init_subclass__(cls, method: str, is_c4: bool, requires: list[str]) -> None:
        """Initialise a subclass deriving from this ABC."""

        cls.method = method
        cls.is_c4 = is_c4
        cls.requires = requires
        OPTIMAL_CHI_CLASS_REGISTRY[cls.method] = cls


class OptimalChiPrentice14(
    OptimalChiABC,
    method="prentice14",
    is_c4=False,
    requires=[],
):
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

    Finally,  :math:`m_c` is calculated, following Equation 7 in :cite:`Stocker:2020dh`,
    as:

    .. math::

        m_c = \frac{c_i - \Gamma^{*}}{c_i + K},

    where :math:`K` is the Michaelis Menten coefficient of Rubisco-limited assimilation.

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]), 
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ... )
        >>> vals = OptimalChiPrentice14(env=env)
        >>> vals.chi.round(5)
        array([0.69435])
        >>> vals.mc.round(5)
        array([0.33408])
        >>> vals.mj.round(5)
        array([0.7123])
        >>> vals.mjoc.round(5)
        array([2.13211])
    """

    def set_beta(self) -> None:
        """Set ``beta`` to a constant C3 specific value."""
        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.beta = self.pmodel_const.beta_cost_ratio_c3

    def estimate_chi(self, xi_values: NDArray[np.float64] | None = None) -> None:
        """Estimate ``chi`` for C3 plants."""

        if xi_values is not None:
            _ = check_input_shapes(self.env.ca, xi_values)
            self.xi = xi_values
        else:
            self.xi = np.sqrt(
                (self.beta * (self.env.kmm + self.env.gammastar))
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


class OptimalChiPrentice14RootzoneStress(
    OptimalChiABC,
    method="prentice14_rootzonestress",
    is_c4=False,
    requires=["rootzonestress"],
):
    r"""Calculate :math:`\chi` for C3 plants following :cite:`Prentice:2014bc` with root
    zone stress penalty.

    The calculations are identical to
    :class:`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14` but apply an
    experimental rootzone stress penalty (:math:`f_{rz}`) to :math:`\beta` in the
    calculation of :math:`\xi`:

    .. math::

        \xi = \sqrt{(\beta f_{rz} (K+ \Gamma^{*}) / (1.6 \eta^{*}))}

    Calculation of the root zone stress factor is not currently part of the `pyrealm`
    package.

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ...     rootzonestress=0.5
        ... )
        >>> vals = OptimalChiPrentice14RootzoneStress(env=env)
        >>> vals.chi.round(5)
        array([0.62016])
    """  # noqa D210, D415 - long but sane title line.

    __experimental__ = True

    def set_beta(self) -> None:
        """Set ``beta`` to a constant C3 specific value."""

        # Warn that this is an experimental feature.
        warn_experimental("OptimalChiPrentice14RootzoneStress")

        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.beta = self.pmodel_const.beta_cost_ratio_c3

    def estimate_chi(self, xi_values: NDArray[np.float64] | None = None) -> None:
        """Estimate ``chi`` for C3 plants."""

        if xi_values is not None:
            _ = check_input_shapes(self.env.ca, xi_values)
            self.xi = xi_values
        else:
            self.xi = np.sqrt(
                (
                    self.beta
                    * getattr(self.env, "rootzonestress")
                    * (self.env.kmm + self.env.gammastar)
                )
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


class OptimalChiC4(
    OptimalChiABC,
    method="c4",
    is_c4=True,
    requires=[],
):
    r"""Estimate :math:`\chi` for C4 plants following :cite:`Prentice:2014bc`.

    Optimal :math:`\chi` is calculated as in
    :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14`, but using a C4
    specific estimate of the unit cost ratio :math:`\beta`, see
    :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`.

    This method  sets :math:`m_j = m_c = m_{joc} = 1.0` to capture the boosted
    :math:`\ce{CO2}` concentrations at the chloropolast in C4 photosynthesis.

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ... )
        >>> vals = OptimalChiC4(env=env)
        >>> vals.chi.round(5)
        array([0.44967])
        >>> vals.mj.round(5)
        array([1.])
        >>> vals.mc.round(5)
        array([1.])
    """

    def set_beta(self) -> None:
        """Set ``beta`` to a constant C4 specific value."""
        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.beta = self.pmodel_const.beta_cost_ratio_c4

    def estimate_chi(self, xi_values: NDArray[np.float64] | None = None) -> None:
        """Estimate ``chi`` for C4 plants, setting ``mj`` and ``mc`` to 1."""
        if xi_values is not None:
            _ = check_input_shapes(self.env.ca, xi_values)
            self.xi = xi_values
        else:
            self.xi = np.sqrt(
                (self.beta * (self.env.kmm + self.env.gammastar))
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


class OptimalChiC4RootzoneStress(
    OptimalChiABC,
    method="c4_rootzonestress",
    is_c4=True,
    requires=["rootzonestress"],
):
    r"""Estimate :math:`\chi` for C4 plants following :cite:`Prentice:2014bc` with
     root zone stress penalty.

    The calculations are identical to
    :class:`~pyrealm.pmodel.optimal_chi.OptimalChiC4` but apply an
    experimental rootzone stress penalty (:math:`f_{rz}`) to :math:`\beta` in the
    calculation of :math:`\xi`:

    .. math::

        \xi = \sqrt{(\beta f_{rz} (K+ \Gamma^{*}) / (1.6 \eta^{*}))}

    Calculation of the root zone stress factor is not currently part of the `pyrealm`
    package.

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ...     rootzonestress=0.5
        ... )
        >>> vals = OptimalChiC4RootzoneStress(env=env)
        >>> vals.chi.round(5)
        array([0.37659])
        >>> vals.mj.round(5)
        array([1.])
        >>> vals.mc.round(5)
        array([1.])
    """  # noqa D210, D415 - long but sane title line.

    def set_beta(self) -> None:
        """Set ``beta`` to a constant C4 specific value."""
        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.beta = self.pmodel_const.beta_cost_ratio_c4

    def estimate_chi(self, xi_values: NDArray[np.float64] | None = None) -> None:
        """Estimate ``chi`` for C4 plants, setting ``mj`` and ``mc`` to 1."""
        if xi_values is not None:
            _ = check_input_shapes(self.env.ca, xi_values)
            self.xi = xi_values
        else:
            self.xi = np.sqrt(
                (
                    self.beta
                    * getattr(self.env, "rootzonestress")
                    * (self.env.kmm + self.env.gammastar)
                )
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


class OptimalChiLavergne20C3(
    OptimalChiABC,
    method="lavergne20_c3",
    is_c4=False,
    requires=["theta"],
):
    r"""Estimate :math:`\chi` for C3 plants using soil moisture corrected :math:`\beta`.

    Optimal :math:`\chi` is calculated using a definition of the unit cost ratio $\beta$
    as a function of soil moisture ($\theta$, m3 m-3), following :cite:`lavergne:2020a`:

    .. math:: :nowrap:

        \[
            \beta = e ^ {b\theta + a},
        \]

    The coefficients are experimentally derived values with defaults taken from
    Figure 6a of :cite:`lavergne:2020a` (:math:`a`,
    :attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_c3`;
    :math:`b`,
    :attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_c3`).

    Values of :math:`\chi` and other predictions are then calculated as in
    :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14`. This method
    requires that `env` includes estimates of :math:`\theta`.

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ...     theta=np.array([0.5])
        ... )
        >>> vals = OptimalChiLavergne20C3(env=env)
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

    def set_beta(self) -> None:
        """Set ``beta`` with soil moisture corrections."""

        # Calculate beta as a function of theta
        self.beta = np.exp(
            self.pmodel_const.lavergne_2020_c3[1] * getattr(self.env, "theta")
            + self.pmodel_const.lavergne_2020_c3[0]
        )

    def estimate_chi(self, xi_values: NDArray[np.float64] | None = None) -> None:
        """Estimate ``chi`` for C3 plants."""

        if xi_values is not None:
            _ = check_input_shapes(self.env.ca, xi_values)
            self.xi = xi_values
        else:
            self.xi = np.sqrt(
                (self.beta * (self.env.kmm + self.env.gammastar))
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


class OptimalChiLavergne20C4(
    OptimalChiABC,
    method="lavergne20_c4",
    is_c4=True,
    requires=["theta"],
):
    r"""Calculate soil moisture corrected :math:`\chi` for C4 plants.

    This method calculates :math:`\beta` as a function of soil moisture following
    the equation described in the
    :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C3` method.
    However,
    the default coefficients of the moisture scaling from :cite:`lavergne:2020a` for
    C3 plants are adjusted to match the theoretical expectation that :math:`\beta`
    for C4 plants is nine times smaller than :math:`\beta` for C3 plants (see
    :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiC4`): :math:`b`
    (:attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_c4`) is
    unchanged but
    :math:`a_{C4} = a_{C3} - log(9)`
    (:attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_c4`) .

    Following the calculation of :math:`\beta`, this method then follows the
    calculations described in
    :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiC4NoGamma`
    ::math:`m_j = 1.0`
    because photorespiration is negligible, but :math:`m_c` and hence
    :math:`m_{joc}` are calculated.

    .. NOTE::

        The research underlying :cite:`lavergne:2020a`, found **no relationship**
        between C4 :math:`\beta` values and soil moisture in leaf gas exchange
        measurements. This implementation asserts that there is a relationship and that
        the relationship is consistent with the patterns found for C3 plants.

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ...     theta=np.array([0.5])
        ... )
        >>> vals = OptimalChiLavergne20C4(env=env)
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

    __experimental__ = True

    def set_beta(self) -> None:
        """Set ``beta`` with soil moisture corrections."""

        warn_experimental("OptimalChiLavergne20C4")

        # Calculate beta as a function of theta
        self.beta = np.exp(
            self.pmodel_const.lavergne_2020_c4[1] * getattr(self.env, "theta")
            + self.pmodel_const.lavergne_2020_c4[0]
        )

    def estimate_chi(self, xi_values: NDArray[np.float64] | None = None) -> None:
        """Estimate ``chi`` for C4 plants excluding photorespiration."""

        # Calculate chi and xi as in Prentice 14 but removing gamma terms.
        if xi_values is not None:
            _ = check_input_shapes(self.env.ca, xi_values)
            self.xi = xi_values
        else:
            self.xi = np.sqrt((self.beta * self.env.kmm) / (1.6 * self.env.ns_star))

        self.chi = self.xi / (self.xi + np.sqrt(self.env.vpd))

        # mj is equal to 1 as gammastar is null
        self.mj = np.ones(self.shape)

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mc = (self.ci) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc


class OptimalChiC4NoGamma(
    OptimalChiABC,
    method="c4_no_gamma",
    is_c4=True,
    requires=[],
):
    r"""Calculate optimal chi for C4 plants assuming negligible photorespiration.

    Calculates :math:`\chi` assuming that photorespiration (:math:`\Gamma^\ast`) is
    negligible for C4 plants. This simplifies the calculation of :math:`\xi` and
    :math:`\chi` compared to :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiC4`,
    but uses the same C4 specific estimate of the unit cost ratio :math:`\beta`,
    :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`.

    .. math:: :nowrap:

        \[
            \begin{align*}
                \chi &= \xi / (\xi + \sqrt D ), \text{where}\\ 
                \xi &= \sqrt{(\beta K) / (1.6 \eta^{*}))}
            \end{align*}
        \]

    In addition, :math:`m_j = 1.0`  because photorespiration is negligible in C4
    photosynthesis, but :math:`m_c` and hence :math:`m_{joc}` are calculated, not
    set to one.

    .. math::

        m_c = \dfrac{\chi}{\chi + \frac{K}{c_a}}

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ... )
        >>> vals = OptimalChiC4NoGamma(env=env)
        >>> vals.chi.round(5)
        array([0.3919])
        >>> vals.mj.round(5)
        array([1.])
        >>> vals.mc.round(5)
        array([0.25626])
    """

    def set_beta(self) -> None:
        """Set constant ``beta`` for C4 plants."""

        # Calculate chi and xi as in Prentice 14 but removing gamma terms.
        self.beta = self.pmodel_const.beta_cost_ratio_c4

    def estimate_chi(self, xi_values: NDArray[np.float64] | None = None) -> None:
        """Estimate ``chi`` for C4 plants excluding photorespiration."""

        # Calculate chi and xi as in Prentice 14 but removing gamma terms.
        if xi_values is not None:
            _ = check_input_shapes(self.env.ca, xi_values)
            self.xi = xi_values
        else:
            self.xi = np.sqrt((self.beta * self.env.kmm) / (1.6 * self.env.ns_star))

        # self.xi = np.sqrt(
        #     (self.beta * self.rootzonestress * self.env.kmm) /
        #     (1.6 * self.env.ns_star)
        # )

        self.chi = self.xi / (self.xi + np.sqrt(self.env.vpd))

        # mj is equal to 1 as gammastar is null
        self.mj = np.ones(self.shape)

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mc = (self.ci) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc


class OptimalChiC4NoGammaRootzoneStress(
    OptimalChiABC,
    method="c4_no_gamma_rootzonestress",
    is_c4=True,
    requires=["rootzonestress"],
):
    r"""Calculate optimal chi for C4 plants assuming negligible photorespiration with
     root zone stress penalty.

    The calculations are identical to
    :class:`~pyrealm.pmodel.optimal_chi.OptimalChiC4NoGamma` but apply an
    experimental rootzone stress penalty (:math:`f_{rz}`) to :math:`\beta` in the
    calculation of :math:`\xi`:

    .. math::

        \xi = \sqrt{(\beta f_{rz} K) / (1.6 \eta^{*}))}

    Calculation of the root zone stress factor is not currently part of the `pyrealm`
    package.

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ...     rootzonestress=np.array([0.5])
        ... )
        >>> vals = OptimalChiC4NoGammaRootzoneStress(env=env)
        >>> vals.chi.round(5)
        array([0.31305])
        >>> vals.mj.round(5)
        array([1.])
        >>> vals.mc.round(5)
        array([0.21583])
    """  # noqa D210, D415 - long but sane title line.

    __experimental__ = True

    def set_beta(self) -> None:
        """Set constant ``beta`` for C4 plants."""

        warn_experimental("OptimalChiC4NoGammaRootzoneStress")

        # Calculate chi and xi as in Prentice 14 but removing gamma terms.
        self.beta = self.pmodel_const.beta_cost_ratio_c4

    def estimate_chi(self, xi_values: NDArray[np.float64] | None = None) -> None:
        """Estimate ``chi`` for C4 plants excluding photorespiration."""

        # Calculate chi and xi as in Prentice 14 but removing gamma terms.
        if xi_values is not None:
            _ = check_input_shapes(self.env.ca, xi_values)
            self.xi = xi_values
        else:
            self.xi = np.sqrt(
                (self.beta * getattr(self.env, "rootzonestress") * self.env.kmm)
                / (1.6 * self.env.ns_star)
            )

        self.chi = self.xi / (self.xi + np.sqrt(self.env.vpd))

        # mj is equal to 1 as gammastar is null
        self.mj = np.ones(self.shape)

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mc = (self.ci) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc
