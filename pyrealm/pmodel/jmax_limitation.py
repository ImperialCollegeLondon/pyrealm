"""The module :mod:`~pyrealm.pmodel.jmax_limitation` provides the implementation of
the following pmodel core class:

* :class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitation`:
    Estimates the Jmax limitation, given a method and settings.

"""  # noqa D210, D415

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import PModelConst
from pyrealm.core.utilities import check_input_shapes
from pyrealm.pmodel.optimal_chi import OptimalChiABC


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
        optchi: an instance of :class:`pyrealm.pmodel.optimal_chi.OptimalChiABC`
            providing the :math:`\ce{CO2}` limitation term of light use efficiency
            (:math:`m_j`) and the :math:`\ce{CO2}` limitation term for Rubisco
            assimilation (:math:`m_c`).
        method: method to apply :math:`J_{max}` limitation (default: ``wang17``,
            or ``smith19`` or ``none``)
        pmodel_const: An instance of
            :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    Examples:
        >>> import numpy as np
        >>> from pyrealm.pmodel.pmodel_environment import PModelEnvironment
        >>> from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0])
        ... )
        >>> optchi = OptimalChiPrentice14(env)
        >>> simple = JmaxLimitation(optchi, method='simple')
        >>> simple.f_j
        array([1.])
        >>> simple.f_v
        array([1.])
        >>> wang17 = JmaxLimitation(optchi, method='wang17')
        >>> wang17.f_j.round(5)
        array([0.66722])
        >>> wang17.f_v.round(5)
        array([0.55502])
        >>> smith19 = JmaxLimitation(optchi, method='smith19')
        >>> smith19.f_j.round(5)
        array([1.10204])
        >>> smith19.f_v.round(5)
        array([0.75442])
    """

    # TODO - apparent incorrectness of wang and smith methods with _ca_ variation,
    #        work well with varying temperature but not _ca_ variation (or
    #        e.g. elevation gradient David Sandoval, REALM meeting, Dec 2020)

    def __init__(
        self,
        optchi: OptimalChiABC,
        method: str = "wang17",
        pmodel_const: PModelConst = PModelConst(),
    ):
        self.shape: tuple = check_input_shapes(optchi.mj)
        """Records the common numpy array shape of array inputs."""
        self.optchi: OptimalChiABC = optchi
        """Details of the optimal chi calculation for the model"""
        self.method: str = method
        """Records the method used to calculate Jmax limitation."""
        self.pmodel_const: PModelConst = pmodel_const
        """The PModelParams instance used for the calculation."""

        # Attributes populated by alternative method - two should always be populated by
        # the methods used below, but omega and omega_star only apply to smith19
        self.f_j: NDArray
        """:math:`J_{max}` limitation factor, calculated using the method."""
        self.f_v: NDArray
        """:math:`V_{cmax}` limitation factor, calculated using the method."""
        self.omega: NDArray | None = None
        """Component of :math:`J_{max}` calculation for method ``smith19``
        (:cite:`Smith:2019dv`)."""
        self.omega_star: NDArray | None = None
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
        and is set in `pmodel_const.wang_c`.
        """

        # Calculate √ {1 - (c*/m)^(2/3)} (see Eqn 2 of Wang et al 2017) and
        # √ {(m/c*)^(2/3) - 1} safely, both are undefined where m <= c*.
        vals_defined = np.greater(self.optchi.mj, self.pmodel_const.wang17_c)

        self.f_v = np.sqrt(
            1 - (self.pmodel_const.wang17_c / self.optchi.mj) ** (2.0 / 3.0),
            where=vals_defined,
        )
        self.f_j = np.sqrt(
            (self.optchi.mj / self.pmodel_const.wang17_c) ** (2.0 / 3.0) - 1,
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
          in the :meth:`~pyrealm.pmodel.jmax_limitation.JmaxLimitation.wang17` method.
        """

        # Adopted from Nick Smith's code:
        # Calculate omega, see Smith et al., 2019 Ecology Letters  # Eq. S4
        theta = self.pmodel_const.smith19_theta
        c_cost = self.pmodel_const.smith19_c_cost

        # simplification terms for omega calculation
        cm = 4 * c_cost / self.optchi.mj
        v = 1 / (cm * (1 - self.pmodel_const.smith19_theta * cm)) - 4 * theta

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
