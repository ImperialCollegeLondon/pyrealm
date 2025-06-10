"""The :mod:`~pyrealm.constants.core_const` module provides a data class of constants
used in the functions in the :mod:`~pyrealm.core` submodules.
"""  # noqa: D205

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class CoreConst(ConstantsClass):
    r"""Constants for the :mod:`~pyrealm.core`  module.

    This data class provides constants used in :mod:`~pyrealm.core`, including
    hygrometric conversions, the calculation of water density and viscosity and the
    calculation of atmospheric pressure.

    Sources for some sets of parameters are shown below:

    * **Density of water using Fisher**. Values for the Tumlirz equation taken from
      Table 5 of :cite:t:`Fisher:1975tm`:

      .. math::

            V_p = V_\infty + \dfrac{\lambda}{P_o + P},

      where :math:`\lambda`
      (:attr:`~pyrealm.constants.core_const.CoreConst.fisher_dial_lambda`),
      :math:`P_o` (:attr:`~pyrealm.constants.core_const.CoreConst.fisher_dial_Po`) and
      :math:`V_\infty`
      (:attr:`~pyrealm.constants.core_const.CoreConst.fisher_dial_Vinf`) are all
      temperature dependent polynomial functions.

    * **Density of water using Chen**. Values taken from :cite:t:`chen:2008a` and
      calculated as the inverse of their equation for the specific volume of water
      (:math:`V`):

        .. math::
            V = V^0 - V^0 P/(K^0 +AP+BP^2),

      where :math:`V^0` (:attr:`~pyrealm.constants.core_const.CoreConst.chen_po`),
      :math:`K^0` (:attr:`~pyrealm.constants.core_const.CoreConst.chen_ko`),
      :math:`A` (:attr:`~pyrealm.constants.core_const.CoreConst.chen_ca`, and
      :math:`B` (:attr:`~pyrealm.constants.core_const.CoreConst.chen_cb`) are all
      temperature dependent polynomial functions.

    * **Viscosity of water**. Values for the parameterisation taken from Table 2 and 3
      of :cite:t:`Huber:2009fy`:
      (:attr:`~pyrealm.constants.core_const.CoreConst.huber_tk_ast`,
      :attr:`~pyrealm.constants.core_const.CoreConst.huber_rho_ast`,
      :attr:`~pyrealm.constants.core_const.CoreConst.huber_mu_ast`,
      :attr:`~pyrealm.constants.core_const.CoreConst.huber_H_i`,
      :attr:`~pyrealm.constants.core_const.CoreConst.huber_H_ij`)

    """

    # Universal constants
    k_R: float = 8.3145
    """Universal gas constant (:math:`R` , 8.3145, J/mol/K)"""
    k_co: float = 209476.0
    """O2 partial pressure, Standard Atmosphere (:math:`co` , 209476.0, ppm)"""
    k_c_molmass: float = 12.0107
    """Molecular mass of carbon (:math:`c_molmass` , 12.0107, g)"""
    k_water_molmass: float = 18.01258
    """Molecular mass of water (:math:`h2o_molmass` , 18.01258, g)"""
    k_Po: float = 101325.0
    """Standard reference atmosphere (Allen, 1973) (:math:`P_o` , 101325.0, Pa)"""
    k_To: float = 298.15
    """Standard reference temperature (:math:`T_o` ,  298.15, K)"""
    k_L: float = 0.0065
    """Adiabiatic temperature lapse rate (Allen, 1973)   (:math:`L` , 0.0065, K/m)"""
    k_G: float = 9.80665
    """Gravitational acceleration (:math:`G` , 9.80665, m/s^2)"""
    k_Ma: float = 0.028963
    """Molecular weight of dry air (Tsilingiris, 2008)  (:math:`M_a`, 0.028963,
    kg/mol)"""
    k_Mv: float = 0.01802
    """Molecular weight of water vapour (Tsilingiris, 2008)  (:math:`M_v`,0.01802,
    kg/mol)"""
    k_CtoK: float = 273.15
    """Conversion from °C to K   (:math:`CtoK` , 273.15, -)"""
    k_pir = np.pi / 180.0
    """Conversion factor from radians to degrees   (``pir`` , ~0.01745, -)"""

    # TODO - these might be better in a separate SplashConst
    k_w = 0.26
    """Entrainment factor (Lhomme, 1997; Priestley & Taylor, 1972)"""
    k_Cw = 1.05
    """Supply constant, mm/hr (Federer, 1982)"""

    # Solar constants
    visible_light_albedo: float = 0.03
    """The visible light albedo (:math:`A_{vis}`, unitless, Sellers, 1985)."""

    swdown_to_ppfd_factor: float = 2.04
    """Conversion factor from shortwave downwelling radiation (W m-2) to photosynthetic
    photon flux density (PPFD, µmol m-2 s-1): one W m-2 of sunlight is roughly 4.57 µmol
    m-2 s-1 of full spectrum sunlight, of which about 46% (2.04 / 4.57) is PPFD. (Meek
    et al., 1984)."""

    transmissivity_coef: tuple[float, float, float] = (0.25, 0.5, 2.67e-5)
    """Coefficients for calculating transmissivity from :cite:t:`Linacre:1968a`: cloudy
    transmissivity (:math:`c=0.25`),  angular coefficient of transmittivity
    (:math:`d=0.5`) and elevation factor (:math:`f`=2.67e-5`)."""

    net_longwave_radiation_coef: tuple[float, float] = (0.2, 107.0)
    """Coefficients (:math:`b, A`) of net longwave radiation function, Eqn. 11 and Table
    1 of :cite:t:`colinprentice:1993a`"""

    shortwave_albedo: float = 0.17
    """The shortwave albedo (:math:`A_{sw}`, unitless, Federer, 1968)."""

    solar_constant: float = 1360.8
    """The solar constant (:math:`G_{sc}`, W/m^2, Kopp & Lean, 2011) - the long term
    mean total solar irradiance."""

    day_seconds: float = 86400
    """The number of seconds in one solar day."""

    equation_of_time_coef: tuple[float, ...] = (
        0.000075,
        0.001868,
        -0.032077,
        -0.014615,
        -0.04089,
        229.18,
    )
    """Coefficients of the equation of time :cite:t:`iqbal:1983a`"""

    # Paleoclimate variables:
    solar_eccentricity: float = 0.0167
    """Solar eccentricity (:math:`e`), using default value for 2000 CE 
    :cite:t:`berger:1978a`."""
    solar_obliquity: float = 23.44
    r"""Solar obliquity in degrees (:math:`\epsilon`), using default value for 2000 CE
    :cite:t:`berger:1978a`."""
    solar_perihelion: float = 283.0
    r"""Solar longitude of perihelion in degrees (:math:`\omega`), using default value
    for 2000 CE :cite:t:`berger:1978a`."""

    # Hygro constants
    magnus_coef: NDArray[np.float64] = field(
        default_factory=lambda: np.array((611.2, 17.62, 243.12))
    )
    """Three coefficients of the Magnus equation for saturated vapour pressure,
    defaulting to those of  ``Sonntag1990``."""
    mwr: float = 0.622
    """The ratio molecular weight of water vapour to dry air (:math:`MW_r`, -)"""
    magnus_option: str | None = None
    """Pre-defined Magnus equation parameterisations. Use one of ``Allen1998``,
    ``Alduchov1996`` or ``Sonntag1990`` to set values for
    :attr:`~pyrealm.constants.core_const.CoreConst.magnus_coef`.
    """

    # Water constants
    water_density_method: str = "fisher"
    """Set the method used for calculating water density ('fisher' or 'chen')."""

    # Fisher Dial
    fisher_dial_lambda: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [1788.316, 21.55053, -0.4695911, 0.003096363, -7.341182e-06]
        )
    )
    r"""Coefficients of the temperature dependent polynomial for :math:`\lambda`
     in the Tumlirz equation."""

    fisher_dial_Po: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [5918.499, 58.05267, -1.1253317, 0.0066123869, -1.4661625e-05]
        )
    )
    """Coefficients of the temperature dependent polynomial for :math:`P_0` in the
    Tumlirz equation."""

    fisher_dial_Vinf: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [
                0.6980547,
                -0.0007435626,
                3.704258e-05,
                -6.315724e-07,
                9.829576e-09,
                -1.197269e-10,
                1.005461e-12,
                -5.437898e-15,
                1.69946e-17,
                -2.295063e-20,
            ]
        )
    )
    r"""Coefficients of the temperature dependent polynomial for :math:`V_{\infty}`
    in the Tumlirz equation."""

    # Chen water density
    chen_po: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [
                0.99983952,
                6.788260e-5,
                -9.08659e-6,
                1.022130e-7,
                -1.35439e-9,
                1.471150e-11,
                -1.11663e-13,
                5.044070e-16,
                -1.00659e-18,
            ]
        )
    )
    r"""Coefficients of the polynomial relationship of water density with temperature at
    1 atm (:math:`P^0`, kg/m^3) from :cite:t:`chen:2008a`."""

    chen_ko: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [19652.17, 148.1830, -2.29995, 0.01281, -4.91564e-5, 1.035530e-7]
        )
    )
    r"""Polynomial relationship of bulk modulus of water with temperature at 1 atm
     (:math:`K^0`, kg/m^3) from :cite:t:`chen:2008a`."""

    chen_ca: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [3.26138, 5.223e-4, 1.324e-4, -7.655e-7, 8.584e-10]
        )
    )
    r"""Coefficients of the polynomial temperature dependent coefficient :math:`A` from
     :cite:t:`chen:2008a`."""

    chen_cb: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [7.2061e-5, -5.8948e-6, 8.69900e-8, -1.0100e-9, 4.3220e-12]
        )
    )
    r"""Coefficients of the polynomial temperature dependent coefficient :math:`B` from
     :cite:t:`chen:2008a`."""

    # Huber
    simple_viscosity: bool = False
    """Boolean setting for use of simple viscosity calculations"""
    huber_tk_ast: float = 647.096
    """Huber reference temperature (:math:`tk_{ast}`, 647.096, Kelvin)"""
    huber_rho_ast: float = 322.0
    r"""Huber reference density (:math:`\rho_{ast}`, 322.0, kg/m^3)"""
    huber_mu_ast: float = 1e-06
    r"""Huber reference pressure (:math:`\mu_{ast}` 1.0e-6, Pa s)"""

    huber_H_i: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.67752, 2.20462, 0.6366564, -0.241605])
    )
    """Temperature dependent parameterisation of Hi in Huber."""
    huber_H_ij: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [
                [0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0],
                [0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573],
                [-0.281378, -0.906851, -0.772479, -0.489837, -0.25704, 0.0],
                [0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0],
                [-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0],
                [0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264],
            ],
        )
    )
    """Temperature and mass density dependent parameterisation of Hij in Huber."""

    def __post_init__(self) -> None:
        """Populate parameters from init settings.

        This checks the init inputs and populates ``magnus_coef`` from the presets
        if no magnus_coef is specified.

        Returns:
            None
        """
        alts = dict(
            Allen1998=np.array((610.8, 17.27, 237.3)),
            Alduchov1996=np.array((610.94, 17.625, 243.04)),
            Sonntag1990=np.array((611.2, 17.62, 243.12)),
        )

        # Note that object is being used here to update a frozen dataclass

        # Parse other options
        if self.magnus_option is not None:
            if self.magnus_option not in alts:
                raise (ValueError(f"magnus_option must be one of {list(alts.keys())}"))

            object.__setattr__(self, "magnus_coef", alts[self.magnus_option])
            return

        if self.magnus_coef is not None and len(self.magnus_coef) != 3:
            raise TypeError("magnus_coef must be a tuple of 3 numbers")
