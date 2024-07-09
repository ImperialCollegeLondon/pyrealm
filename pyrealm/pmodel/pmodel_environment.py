"""The module :mod:`~pyrealm.pmodel.pmodel_environment` provides the implementation of
the following pmodel core class:

* :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`:
    Calculates the photosynthetic environment for locations.
"""  # noqa D210, D415

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst, PModelConst
from pyrealm.core.utilities import bounds_checker, check_input_shapes, summarize_attrs
from pyrealm.pmodel.functions import (
    calc_co2_to_ca,
    calc_gammastar,
    calc_kmm,
    calc_ns_star,
)


class PModelEnvironment:
    r"""Create a PModelEnvironment instance.

    This class takes the four key environmental inputs to the P Model and
    calculates four photosynthetic variables for those environmental
    conditions:

    * the photorespiratory :math:`\ce{CO2}` compensation point (:math:`\Gamma^{*}`,
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

    * the volumetric soil moisture content (:math:`\theta`), required to calculate
      optimal :math:`\chi` in
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C3` and
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C3`.

    * a unitless root zone stress factor, an experimental term used to optionally
      penalise the :math:`\beta` term in the estimation of :math:`\chi` in
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14RootzoneStress` and
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiC4RootzoneStress` and
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiC4NoGammaRootzoneStress`.

    Args:
        tc: Temperature, relevant for photosynthesis (°C)
        vpd: Vapour pressure deficit (Pa)
        co2: Atmospheric :math:`\ce{CO2}` concentration (ppm)
        patm: Atmospheric pressure (Pa)
        theta: Volumetric soil moisture (m3/m3)
        rootzonestress: Root zone stress factor (-)
        pmodel_const: An instance of
            :class:`~pyrealm.constants.pmodel_const.PModelConst`.
        core_const: An instance of
            :class:`~pyrealm.constants.core_const.CoreConst`.

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0])
        ... )
    """

    def __init__(
        self,
        tc: NDArray,
        vpd: NDArray,
        co2: NDArray,
        patm: NDArray,
        theta: NDArray | None = None,
        rootzonestress: NDArray | None = None,
        pmodel_const: PModelConst = PModelConst(),
        core_const: CoreConst = CoreConst(),
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
        if np.nanmin(self.tc) < np.array([-25]):
            raise ValueError(
                "Cannot calculate P Model predictions for values below"
                " -25°C. See calc_density_h2o."
            )

        # Guard against negative VPD issues
        if np.nanmin(self.vpd) < np.array([0]):
            raise ValueError(
                "Negative VPD values will lead to missing data - clip to "
                "zero or explicitly set to np.nan"
            )

        self.ca: NDArray = calc_co2_to_ca(self.co2, self.patm)
        """Ambient CO2 partial pressure, Pa"""

        self.gammastar = calc_gammastar(
            tc, patm, pmodel_const=pmodel_const, core_const=core_const
        )
        r"""Photorespiratory compensation point (:math:`\Gamma^\ast`, Pa)"""

        self.kmm = calc_kmm(tc, patm, pmodel_const=pmodel_const, core_const=core_const)
        """Michaelis Menten coefficient, Pa"""

        # # Michaelis-Menten coef. C4 plants (Pa) NOT CHECKED. Need to think
        # # about how many optional variables stack up in PModelEnvironment
        # # and this is only required by C4 optimal chi Scott and Smith, which
        # # has not yet been implemented.
        # self.kp_c4 = calc_kp_c4(tc, patm, const=const)

        self.ns_star = calc_ns_star(tc, patm, core_const=core_const)
        """Viscosity correction factor realtive to standard
        temperature and pressure, unitless"""

        # Optional variables
        self.theta: NDArray | None = None
        """Volumetric soil moisture (m3/m3)"""
        self.rootzonestress: NDArray | None = None
        """Rootzone stress factor (experimental) (-)"""

        if theta is not None:
            # Is the input congruent with the other variables and in bounds.
            _ = check_input_shapes(tc, theta)
            self.theta = bounds_checker(theta, 0, 0.8, "[]", "theta", "m3/m3")

        if rootzonestress is not None:
            # Is the input congruent with the other variables and in bounds.
            _ = check_input_shapes(tc, rootzonestress)
            self.rootzonestress = bounds_checker(
                rootzonestress, 0, 1, "[]", "rootzonestress", "-"
            )

        # Store constant settings
        self.pmodel_const = pmodel_const
        """PModel constants used to calculate environment"""
        self.core_const = core_const
        """Core constants used to calculate environment"""

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
