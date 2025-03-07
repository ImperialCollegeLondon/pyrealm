"""The :mod:`~pyrealm.core.hygro` submodule provides conversion functions for
common hygrometric variables. The module provides conversions to vapour pressure
deficit, which is the required input for the
:class:`~pyrealm.pmodel.pmodel.PModel` from vapour pressure, specific humidity
and relative humidity. The implementation is drawn from and validated against the
``bigleaf`` R package.
"""  # noqa: D205

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.bounds import BoundsChecker
from pyrealm.core.utilities import evaluate_horner_polynomial


def calc_vp_sat(
    ta: NDArray[np.float64], core_const: CoreConst = CoreConst()
) -> NDArray[np.float64]:
    r"""Calculate vapour pressure of saturated air.

    This function calculates the vapour pressure of saturated air in kPa at a given
    temperature in °C, using the Magnus equation:

    .. math::

        P = a \exp\(\frac{b - T}{T + c}\),

    where :math:`a,b,c` are defined in
    :attr:`~pyrealm.constants.core_const.CoreConst.magnus_coef`.

    Args:
        ta: The air temperature in °C.
        core_const: An instance of :class:`~pyrealm.constants.core_const.CoreConst`
            giving the parameters for conversions.

    Returns:
        Saturated air vapour pressure in kPa.

    Examples:
        >>> # Saturated vapour pressure at 21°C
        >>> import numpy as np
        >>> temp = np.array([21])
        >>> calc_vp_sat(temp).round(6)
        array([2.480904])
        >>> from pyrealm.constants import CoreConst
        >>> allen = CoreConst(magnus_option='Allen1998')
        >>> calc_vp_sat(temp, core_const=allen).round(6)
        array([2.487005])
        >>> alduchov = CoreConst(magnus_option='Alduchov1996')
        >>> calc_vp_sat(temp, core_const=alduchov).round(6)
        array([2.481888])
    """

    # Magnus equation and conversion to kPa
    cf = core_const.magnus_coef
    vp_sat = cf[0] * np.exp((cf[1] * ta) / (cf[2] + ta)) / 1000

    return vp_sat


def convert_vp_to_vpd(
    vp: NDArray[np.float64],
    ta: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    """Convert vapour pressure to vapour pressure deficit.

    Args:
        vp: The vapour pressure in kPa
        ta: The air temperature in °C
        core_const: An instance of :class:`~pyrealm.constants.core_const.CoreConst`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> import numpy as np
        >>> from pyrealm.constants import CoreConst
        >>> vp = np.array([1.9])
        >>> temp = np.array([21])
        >>> convert_vp_to_vpd(vp, temp).round(7)
        array([0.5809042])
        >>> allen = CoreConst(magnus_option='Allen1998')
        >>> convert_vp_to_vpd(vp, temp, core_const=allen).round(7)
        array([0.5870054])
    """
    vp_sat = calc_vp_sat(ta, core_const=core_const)

    return vp_sat - vp


def convert_rh_to_vpd(
    rh: NDArray[np.float64],
    ta: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
    bounds_checker: BoundsChecker = BoundsChecker(),
) -> NDArray[np.float64]:
    """Convert relative humidity to vapour pressure deficit.

    Args:
        rh: The relative humidity (proportion in (0,1))
        ta: The air temperature in °C
        core_const: An instance of :class:`~pyrealm.constants.core_const.CoreConst`
            giving the settings to be used in conversions.
        bounds_checker: A BoundsChecker instance used to validate inputs.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> import numpy as np
        >>> from pyrealm.constants import CoreConst
        >>> import sys; sys.stderr = sys.stdout
        >>> rh = np.array([0.7])
        >>> temp = np.array([21])
        >>> convert_rh_to_vpd(rh, temp).round(7)
        array([0.7442712])
        >>> allen = CoreConst(magnus_option='Allen1998')
        >>> convert_rh_to_vpd(rh, temp, core_const=allen).round(7)
        array([0.7461016])
        >>> rh_percent = np.array([70])
        >>> convert_rh_to_vpd(rh_percent, temp).round(7) #doctest: +ELLIPSIS
        pyrealm... contains values outside the expected range (0,1). Check units?
        array([-171.1823864])
    """

    rh = bounds_checker.check("rh", rh)

    vp_sat = calc_vp_sat(ta, core_const=core_const)

    return vp_sat - (rh * vp_sat)


def convert_sh_to_vp(
    sh: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    """Convert specific humidity to vapour pressure.

    Args:
        sh: The specific humidity in kg kg-1
        patm: The atmospheric pressure in kPa
        core_const: An instance of :class:`~pyrealm.constants.core_const.CoreConst`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure in kPa

    Examples:
        >>> import numpy as np
        >>> sh = np.array([0.006])
        >>> patm = np.array([99.024])
        >>> convert_sh_to_vp(sh, patm).round(7)
        array([0.9517451])
    """

    return sh * patm / ((1.0 - core_const.mwr) * sh + core_const.mwr)


def convert_sh_to_vpd(
    sh: NDArray[np.float64],
    ta: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    """Convert specific humidity to vapour pressure deficit.

    Args:
        sh: The specific humidity in kg kg-1
        ta: The air temperature in °C
        patm: The atmospheric pressure in kPa
        core_const: An instance of :class:`~pyrealm.constants.core_const.CoreConst`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> import numpy as np
        >>> from pyrealm.constants import CoreConst
        >>> sh = np.array([0.006])
        >>> temp = np.array([21])
        >>> patm = np.array([99.024])
        >>> convert_sh_to_vpd(sh, temp, patm).round(6)
        array([1.529159])
        >>> allen = CoreConst(magnus_option='Allen1998')
        >>> convert_sh_to_vpd(sh, temp, patm, core_const=allen).round(5)
        array([1.53526])
    """

    vp_sat = calc_vp_sat(ta, core_const=core_const)
    vp = convert_sh_to_vp(sh, patm, core_const=core_const)

    return vp_sat - vp


# The following functions are integrated from the evap.py implementation of SPLASH v1.


def calc_saturation_vapour_pressure_slope(
    tc: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the slope of the saturation vapour pressure curve.

    Calculates the slope of the saturation pressure temperature curve, following
    equation 13 of :cite:t:`allen:1998a`.

    Args:
        tc: The air temperature (°C)

    Returns:
        The calculated slope in kPa °C-1.
    """

    # TODO move these coefficients into constants?
    return (
        17.269
        * 237.3
        * 610.78
        * (np.exp(tc * 17.269 / (tc + 237.3)) / ((tc + 237.3) ** 2))
    )


def calc_enthalpy_vaporisation(tc: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the enthalpy of vaporization.

    Calculates the latent heat of vaporization of water as a function of
    temperature following :cite:t:`henderson-sellers:1984a`.

    Args:
        tc: Air temperature (°C)

    Returns:
        Calculated latent heat of vaporisation (J/Kg).
    """

    # TODO move these coefficients into constants?
    return 1.91846e6 * ((tc + 273.15) / (tc + 273.15 - 33.91)) ** 2


def calc_specific_heat(tc: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the specific heat of air.

    Calculates the specific heat of air at a constant pressure (:math:`c_{pm}`, J/kg/K)
    following :cite:t:`tsilingiris:2008a`. This equation is only valid for temperatures
    between 0 and 100 °C.

    Args:
        tc: Air temperature (°C)

    Returns:
        The specific heat of air values.
    """

    # TODO move these coefficients into constants?

    tc = np.clip(tc, 0, 100)
    cp = 1e3 * evaluate_horner_polynomial(
        tc,
        [
            1.0045714270,
            2.050632750e-3,
            -1.631537093e-4,
            6.212300300e-6,
            -8.830478888e-8,
            5.071307038e-10,
        ],
    )

    return cp


def calc_psychrometric_constant(
    tc: NDArray[np.float64], p: NDArray[np.float64], core_const: CoreConst = CoreConst()
) -> NDArray[np.float64]:
    r"""Calculate the psychrometric constant.

    Calculates the psychrometric constant (:math:`\lambda`, Pa/K) given the temperature
    and atmospheric pressure following :cite:t:`allen:1998a` and
    :cite:t:`tsilingiris:2008a`.

    Args:
        tc: Air temperature (°C)
        p: Atmospheric pressure (Pa)
        core_const: An instance of :class:`~pyrealm.constants.core_const.CoreConst`
            giving the settings to be used in conversions.

    Returns:
        The calculated psychrometric constant
    """

    # Calculate the specific heat capacity of water, J/kg/K
    cp = calc_specific_heat(tc)

    # Calculate latent heat of vaporization, J/kg
    lv = calc_enthalpy_vaporisation(tc)

    # Calculate psychrometric constant, Pa/K
    # Eq. 8, Allen et al. (1998)
    return cp * core_const.k_Ma * p / (core_const.k_Mv * lv)
