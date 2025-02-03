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

    This class takes the temperature (°C), vapour pressure deficit (Pa), atmospheric
    pressure (Pa) and ambient CO2 concentration (ppm) and uses these four drivers to
    calculates four photosynthetic variables for those environmental conditions:

    * the photorespiratory :math:`\ce{CO2}` compensation point (:math:`\Gamma^{*}`,
      using :func:`~pyrealm.pmodel.functions.calc_gammastar`),
    * the relative viscosity of water (:math:`\eta^*`,
      using :func:`~pyrealm.pmodel.functions.calc_ns_star`),
    * the ambient partial pressure of :math:`\ce{CO2}` (:math:`c_a`,
      using :func:`~pyrealm.pmodel.functions.calc_co2_to_ca`) and
    * the Michaelis Menten coefficient of Rubisco-limited assimilation
      (:math:`K`, using :func:`~pyrealm.pmodel.functions.calc_kmm`).

    It also takes the photosynthetic photon flux density (PPFD, µmol m-2 s-1) and the
    fraction of absorbed photosynthetically active radiation (FAPAR, unitless), which is
    used to scale light use efficiency up to gross primary productivity.

    An instance of ``PModelEnvironment`` can then be used to fit different P Models
    using the same environment but different method implementations. Note that the
    underlying constants of the P Model
    (:class:`~pyrealm.constants.pmodel_const.PModelConst`) are set when creating an
    instance of this class and will be used on all models fitted to the instance.

    In addition to the main forcing variables above, the ``PModelEnvironment`` class can
    also be used to provide additional variables used by some methods. Any additional
    arguments provided to a ``PModelEnvironment`` instance should provide an array of
    data congruent with the shape of the other forcing variables. The array dimensions
    will be checked and the argument name will be used to add the data to the instance
    as an additional attribute.

    Examples of additional variables include:

    * the volumetric soil moisture content (:math:`\theta`), required to calculate
      optimal :math:`\chi` in
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C3` and
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C3`.

    * the experimental `rootzonestress` factor used to penalise the :math:`\beta` term
      in the estimation of :math:`\chi` in
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14RootzoneStress` and
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiC4RootzoneStress` and
      :meth:`~pyrealm.pmodel.optimal_chi.OptimalChiC4NoGammaRootzoneStress`.

    * The climatological aridity index, expressed as PET/P (-).

    * The mean growth temperature, calculated as the mean temperature > 0°C during
      growing degree days (°C).

    Args:
        tc: Temperature, relevant for photosynthesis (°C)
        vpd: Vapour pressure deficit (Pa)
        co2: Atmospheric :math:`\ce{CO2}` concentration (ppm)
        patm: Atmospheric pressure (Pa)
        theta: Volumetric soil moisture (m3/m3)
        rootzonestress: Root zone stress factor (-)
        aridity_index: Climatological aridity index, expressed as PET/P (-)
        mean_growth_temperature: Mean growth temperature (°C)
        pmodel_const: An instance of
            :class:`~pyrealm.constants.pmodel_const.PModelConst`.
        core_const: An instance of
            :class:`~pyrealm.constants.core_const.CoreConst`.
        **kwargs: Additional data variables

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]), vpd=np.array([1000]),
        ...     co2=np.array([400]), patm=np.array([101325.0])
        ... )
    """

    def __init__(
        self,
        tc: NDArray[np.float64],
        vpd: NDArray[np.float64],
        co2: NDArray[np.float64],
        patm: NDArray[np.float64],
        fapar: NDArray[np.float64],
        ppfd: NDArray[np.float64],
        pmodel_const: PModelConst = PModelConst(),
        core_const: CoreConst = CoreConst(),
        **kwargs: NDArray[np.float64],
    ):
        # Check shapes of inputs are congruent
        self.shape: tuple = check_input_shapes(
            tc, vpd, co2, patm, fapar, ppfd, **kwargs
        )

        # Validate and store the forcing variables
        self.tc: NDArray[np.float64] = bounds_checker(tc, -25, 80, "[]", "tc", "°C")
        """The temperature at which to estimate photosynthesis, °C"""
        self.vpd: NDArray[np.float64] = bounds_checker(vpd, 0, 10000, "[]", "vpd", "Pa")
        """Vapour pressure deficit, Pa"""
        self.co2: NDArray[np.float64] = bounds_checker(co2, 0, 1000, "[]", "co2", "ppm")
        """CO2 concentration, ppm"""
        self.patm: NDArray[np.float64] = bounds_checker(
            patm, 30000, 110000, "[]", "patm", "Pa"
        )
        """Atmospheric pressure, Pa"""
        self.fapar: NDArray[np.float64] = bounds_checker(
            fapar, 0, 1, "[]", "fapar", "-"
        )
        """The fraction of absorbed photosynthetically active radiation (FAPAR, -)"""
        self.ppfd: NDArray[np.float64] = bounds_checker(
            ppfd, 0, 3000, "[]", "ppfd", "-"
        )
        """The photosynthetic photon flux density (PPFD, µmol m-2 s-1)"""

        # Store constant settings
        self.pmodel_const = pmodel_const
        """PModel constants used to calculate environment"""
        self.core_const = core_const
        """Core constants used to calculate environment"""

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

        # Internal calculations
        self.tk = self.tc + self.core_const.k_CtoK
        """The temperature at which to estimate photosynthesis in Kelvin (K)"""

        # TODO - swap functions over to using tk as needed

        self.ca: NDArray[np.float64] = calc_co2_to_ca(self.co2, self.patm)
        """Ambient CO2 partial pressure, Pa"""

        self.gammastar = calc_gammastar(
            tc, patm, pmodel_const=pmodel_const, core_const=core_const
        )
        r"""Photorespiratory compensation point (:math:`\Gamma^\ast`, Pa)"""

        self.kmm = calc_kmm(tc, patm, pmodel_const=pmodel_const, core_const=core_const)
        """Michaelis Menten coefficient, Pa"""

        self.ns_star = calc_ns_star(tc, patm, core_const=core_const)
        """Viscosity correction factor realtive to standard
        temperature and pressure, unitless"""

        # Additional variables
        for var_name, var_values in kwargs.items():
            setattr(self, var_name, var_values)

        self._additional_var_names: tuple(str) = tuple(kwargs.keys())
        """A tuple containing the attribute names of additional variables passed to the
        PModelEnivronment."""

        # self.theta: NDArray[np.float64]
        # """Volumetric soil moisture (m3/m3)"""
        # self.rootzonestress: NDArray[np.float64]
        # """Rootzone stress factor (experimental) (-)"""
        # self.aridity_index: NDArray[np.float64]
        # """Climatological aridity index as PET/P (-)"""
        # self.mean_growth_temperature: NDArray[np.float64]
        # """Mean temperature > 0°C during growing degree days (°C)"""

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

        attrs: tuple[tuple[str, str], ...] = (
            ("tc", "°C"),
            ("vpd", "Pa"),
            ("co2", "ppm"),
            ("patm", "Pa"),
            ("fapar", "-"),
            ("ppfd", "µmol m-2 s-1"),
            ("ca", "Pa"),
            ("gammastar", "Pa"),
            ("kmm", "Pa"),
            ("ns_star", "-"),
            *self._additional_var_names,
        )

        # Add any optional variables - need to check here if these attributes actually
        # exist because if they are not provided they are typed but not populated.
        optional_vars = [
            ("theta", "m3/m3"),
            ("rootzonestress", "-"),
            ("aridity_index", "-"),
            ("mean_growth_temperature", "°C"),
        ]

        to_add = [
            (opt_var, unit)
            for opt_var, unit in optional_vars
            if getattr(self, opt_var, None) is not None
        ]

        attrs = (*attrs, *to_add)

        summarize_attrs(self, attrs, dp=dp)
