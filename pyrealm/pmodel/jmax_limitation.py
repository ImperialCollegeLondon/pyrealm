"""The :mod:`~pyrealm.pmodel.jmax_limitation` module provides the implementation of
classes for calculation of :math:`J_{max}` and :math:`V_{cmax}` limitation. The module
provides an abstract base dataclass
(:class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationABC`) which provides the core
functionality for the implementation. Individual methods then are defined as subclasses
that only need to add any additional data attributes and define the private
:meth:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationABC._calculate_limitation_terms`
method. This is automatically called by the ``__post_init__`` method of the data class and
so the limitation terms are calculated when an instance is created.

The module defines a registry
(:data:`~pyrealm.pmodel.jmax_limitation.JMAX_LIMITATION_CLASS_REGISTRY`) to track
defined subclasses. Subclasses are added to this dictionary, under a string set by the
subclass ``method`` attribute, by the ``__init_subclass`` method of the base class,
which allows implementations to be selected by a simple string method name.
"""  # noqa D210, D415

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import PModelConst
from pyrealm.core.utilities import summarize_attrs
from pyrealm.pmodel.optimal_chi import OptimalChiABC

JMAX_LIMITATION_CLASS_REGISTRY: dict[str, type[JmaxLimitationABC]] = {}
"""A registry for subclasses of
:class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationABC`. Subclasses are automatically
included in this registry dictionary under their defined ``method`` name.
"""


@dataclass
class JmaxLimitationABC(metaclass=ABCMeta):
    r"""An abstract base class for JMaxLimitation implementations.

    This base class defines the ``__init__`` arguments, common data attributes and core
    methods for implementing JMaxLimitation methods. Subclasses should only need to
    define any additional data attributes that should be exposed to users and define the
    private
    :meth:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationABC._calculate_limitation_terms`
    method for the implementation. Subclass definitions should use
    ``@dataclass(repr=False)`` to avoid overriding the base implementation of the
    ``_repr__`` method, and also need to provide a method name string and a tuple of the
    data attributes to include when the
    :meth:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationABC.summarize` method is called
    for the subclass.

    See :class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationWang17` for an example.
    """

    method: ClassVar[str]
    """A short name for the method of Jmax limitation implemented in the subclass."""
    data_attrs: ClassVar[tuple[tuple[str, str], ...]]
    """A tuple of names and units for the data attributes of the class to be reported 
    by summarize."""
    optchi: OptimalChiABC
    """The optimal chi instance used to calculate limitation terms."""
    pmodel_const: PModelConst = field(default_factory=lambda: PModelConst())
    """The PModel constants instance used for the calculation."""
    _shape: tuple[int, ...] = field(init=False)
    """Records the common numpy array shape in the data."""
    f_j: NDArray[np.float64] = field(init=False)
    """:math:`J_{max}` limitation factor."""
    f_v: NDArray[np.float64] = field(init=False)
    """:math:`V_{cmax}` limitation factor."""

    def __post_init__(self) -> None:
        self._shape = self.optchi.mj.shape

        self._calculate_limitation_terms()

    def __repr__(self) -> str:
        """Generates a string representation of a JmaxLimitation instance."""
        return f"JmaxLimitation(method={self.method}, shape={self._shape})"

    def summarize(self, dp: int = 2) -> None:
        """Print OptimalChi summary.

        Prints a summary of the variables calculated within an instance
        of OptimalChi including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        summarize_attrs(self, self.data_attrs, dp=dp)

    @abstractmethod
    def _calculate_limitation_terms(self) -> None:
        """Abstract method defined in subclasses to populate limitation attributes."""
        pass

    @classmethod
    def __init_subclass__(
        cls, method: str, data_attrs: tuple[tuple[str, str], ...]
    ) -> None:
        """Initialise a subclass deriving from this ABC."""

        cls.method = method
        cls.data_attrs = data_attrs
        JMAX_LIMITATION_CLASS_REGISTRY[cls.method] = cls


@dataclass(repr=False)
class JmaxLimitationWang17(
    JmaxLimitationABC,
    method="wang17",
    data_attrs=(("f_j", "-"), ("f_v", "-")),
):
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
    and is set in
    :attr:`PModelConsts.wang17_c<pyrealm.constants.PModelConst.wang17_c>`. 
    Note that both equations are undefined where :math:`m \le c^*`: where this
    condition is true, values will be returned as ``np.nan``.

    Examples:
        >>> from pyrealm.pmodel import PModelEnvironment
        >>> from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ... )
        >>> optchi = OptimalChiPrentice14(env=env)
        >>> jmaxlim = JmaxLimitationWang17(optchi=optchi)
        >>> jmaxlim.f_j.round(4)
        array([0.6672])
        >>> jmaxlim.f_v.round(4)
        array([0.555])
    """

    def _calculate_limitation_terms(self) -> None:
        """Limitation calculations for the ``wang17`` method."""
        # Test for m > c*
        vals_defined = np.greater(self.optchi.mj, self.pmodel_const.wang17_c)

        self.f_v = np.sqrt(
            1 - (self.pmodel_const.wang17_c / self.optchi.mj) ** (2.0 / 3.0),
            where=vals_defined,
        )
        self.f_j = np.sqrt(
            (self.optchi.mj / self.pmodel_const.wang17_c) ** (2.0 / 3.0) - 1,
            where=vals_defined,
        )

        # Backfill undefined values
        self.f_j[np.logical_not(vals_defined)] = np.nan
        self.f_v[np.logical_not(vals_defined)] = np.nan


@dataclass(repr=False)
class JmaxLimitationSmith19(
    JmaxLimitationABC,
    method="smith19",
    data_attrs=(("f_j", "-"), ("f_v", "-"), ("omega", "-"), ("omega_star", "-")),
):
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
        in the :class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationWang17` limitation
        terms.

    Examples:
        >>> from pyrealm.pmodel import PModelEnvironment
        >>> from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0]),
        ...     fapar=np.array([1]), ppfd=np.array([800]),
        ... )
        >>> optchi = OptimalChiPrentice14(env=env)
        >>> jmaxlim = JmaxLimitationSmith19(optchi=optchi)
        >>> jmaxlim.f_j.round(4)
        array([1.102])
        >>> jmaxlim.f_v.round(4)
        array([0.7544])
    """

    omega: NDArray[np.float64] = field(init=False)
    """Values of the `omega` parameter (:cite:`Smith:2019dv`)."""
    omega_star: NDArray[np.float64] = field(init=False)
    """Values of the `omega_star` parameter (:cite:`Smith:2019dv`)."""

    def _calculate_limitation_terms(self) -> None:
        """Limitation calculations for the ``smith19`` method."""

        # Calculate omega, see Smith et al., 2019 Ecology Letters  # Eq. S4
        theta, c_cost = self.pmodel_const.smith19_coef

        # simplification terms for omega calculation
        cm = 4 * c_cost / self.optchi.mj
        v = 1 / (cm * (1 - theta * cm)) - 4 * theta

        # account for non-linearities at low m values. This code finds
        # the roots of a quadratic function that is defined purely from
        # the scalar theta, so will always be a scalar. The first root
        # is then used to set a filter for calculating omega.

        cap_p = (((1 / 1.4) - 0.7) ** 2 / (1 - theta)) + 3.4
        aquad = -1
        bquad = cap_p
        cquad = -(cap_p * theta)
        roots = np.polynomial.polynomial.polyroots([aquad, bquad, cquad])  # type: ignore [no-untyped-call]

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
        self.f_v = self.omega_star / (2.0 * theta)
        self.f_j = self.omega


@dataclass(repr=True)
class JmaxLimitationNone(
    JmaxLimitationABC, method="none", data_attrs=(("f_j", "-"), ("f_v", "-"))
):
    """No limitation of :math:`J_{max}` and :math:`V_{cmax}`.

    This implementation simply sets :math:`f_v = f_j = 1` to remove any :math:`J_{max}`
    and :math:`V_{cmax}` limitation.
    """

    def _calculate_limitation_terms(self) -> None:
        """Set limitation terms to one."""

        # Set limitation terms to unity
        self.f_v = np.ones(self._shape)
        self.f_j = np.ones(self._shape)
