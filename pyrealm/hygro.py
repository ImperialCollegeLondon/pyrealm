"""The :mod:`~pyrealm.hygro` module provides provides conversion functions for common
hygrometric variables. The module provides conversions to vapour pressure deficit, which
is the required input for the :class:`~pyrealm.pmodel.PModel` from vapour pressure,
specific humidity and relative humidity. The implementation is drawn from and validated
against the ``bigleaf`` R package.
"""  # noqa: D205, D415

import numpy as np
from numpy.typing import NDArray

from pyrealm.bounds_checker import bounds_checker
from pyrealm.param_classes import HygroParams


def calc_vp_sat(ta: NDArray, hygro_params: HygroParams = HygroParams()) -> NDArray:
    r"""Calculate vapour pressure of saturated air.

    This function calculates the vapour pressure of saturated air at a given
    temperature in kPa, using the Magnus equation:

    .. math::

        P = a \exp\(\frac{b - T}{T + c}\),

    where :math:`a,b,c` are defined in :class:`~pyrealm.param_classes.HygroParams`.

    Args:
        ta: The air temperature
        hygro_params: An object of :class:`~pyrealm.param_classes.HygroParams`
            giving the parameters for conversions.

    Returns:
        Saturated air vapour pressure in kPa.

    Examples:
        >>> # Saturated vapour pressure at 21°C
        >>> import numpy as np
        >>> temp = np.array([21])
        >>> round(calc_vp_sat(temp), 6)
        2.480904
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(calc_vp_sat(temp, hygro_params=allen), 6)
        2.487005
        >>> alduchov = HygroParams(magnus_option='Alduchov1996')
        >>> round(calc_vp_sat(temp, hygro_params=alduchov), 6)
        2.481888
    """

    # Magnus equation and conversion to kPa
    cf = hygro_params.magnus_coef
    vp_sat = cf[0] * np.exp((cf[1] * ta) / (cf[2] + ta)) / 1000

    return vp_sat


def convert_vp_to_vpd(
    vp: NDArray, ta: NDArray, hygro_params: HygroParams = HygroParams()
) -> NDArray:
    """Convert vapour pressure to vapour pressure deficit.

    Args:
        vp: The vapour pressure in kPa
        ta: The air temperature in °C
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> import numpy as np
        >>> vp = np.array([1.9])
        >>> temp = np.array([21])
        >>> round(convert_vp_to_vpd(vp, temp), 7)
        0.5809042
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(convert_vp_to_vpd(vp, temp, hygro_params=allen), 7)
        0.5870054
    """
    vp_sat = calc_vp_sat(ta, hygro_params=hygro_params)

    return vp_sat - vp


def convert_rh_to_vpd(
    rh: NDArray, ta: NDArray, hygro_params: HygroParams = HygroParams()
) -> NDArray:
    """Convert relative humidity to vapour pressure deficit.

    Args:
        rh: The relative humidity (proportion in (0,1))
        ta: The air temperature in °C
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> import numpy as np
        >>> rh = np.array([0.7])
        >>> temp = np.array([21])
        >>> round(convert_rh_to_vpd(rh, temp), 7)
        0.7442712
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(convert_rh_to_vpd(rh, temp, hygro_params=allen), 7)
        0.7461016
        >>> import sys; sys.stderr = sys.stdout
        >>> rh_percent = np.array([70])
        >>> round(convert_rh_to_vpd(rh_percent, temp), 7) #doctest: +ELLIPSIS
        pyrealm... contains values outside the expected range (0,1). Check units?
        -171.1823864
    """

    rh = bounds_checker(rh, 0, 1, "[]", "rh", "proportion")

    vp_sat = calc_vp_sat(ta, hygro_params=hygro_params)

    return vp_sat - (rh * vp_sat)


def convert_sh_to_vp(
    sh: NDArray, patm: NDArray, hygro_params: HygroParams = HygroParams()
) -> NDArray:
    """Convert specific humidity to vapour pressure.

    Args:
        sh: The specific humidity in kg kg-1
        patm: The atmospheric pressure in kPa
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure in kPa

    Examples:
        >>> import numpy as np
        >>> sh = np.array([0.006])
        >>> patm = np.array([99.024])
        >>> round(convert_sh_to_vp(sh, patm), 7)
        0.9517451
    """

    return sh * patm / ((1.0 - hygro_params.mwr) * sh + hygro_params.mwr)


def convert_sh_to_vpd(
    sh: NDArray, ta: NDArray, patm: NDArray, hygro_params: HygroParams = HygroParams()
) -> NDArray:
    """Convert specific humidity to vapour pressure deficit.

    Args:
        sh: The specific humidity in kg kg-1
        ta: The air temperature in °C
        patm: The atmospheric pressure in kPa
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> import numpy as np
        >>> sh = np.array([0.006])
        >>> temp = np.array([21])
        >>> patm = np.array([99.024])
        >>> round(convert_sh_to_vpd(sh, temp, patm), 6)
        1.529159
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(convert_sh_to_vpd(sh, temp, patm, hygro_params=allen), 5)
        1.53526
    """

    vp_sat = calc_vp_sat(ta, hygro_params=hygro_params)
    vp = convert_sh_to_vp(sh, patm, hygro_params=hygro_params)

    return vp_sat - vp
