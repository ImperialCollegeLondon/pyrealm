import numpy as np
from pyrealm.param_classes import UtilParams
from pyrealm.constrained_array import ConstraintFactory

"""
This module provides utility functions shared by modules or providing
extra functions such as conversions for common forcing variable inputs, 
such as hygrometric and radiation conversions.
"""


# Psychrometric conversions to VPD for vapour pressure, specific humidity and
# relative humidity. Using the bigleaf R package as a checking reference from
# which the doctest values are taken

_constrain_rh = ConstraintFactory(lower=0, upper=1, label='relative humidity (-)',)


def calc_vp_sat(ta, util_params=UtilParams()):

    """
    Calculates the vapour pressure of saturated air at a given temperature
    in kPa, using the Magnus equation:

    .. math::

        P = a \exp\(\frac{b - T}{T + c}\)

    The parameters :math:`a,b,c` can provided as a tuple, but three
    built-in options can be selected using a string.

    * ``Allen1998``: (610.8, 17.27, 237.3)
    * ``Alduchov1996``: (610.94, 17.625, 243.04)
    * ``Sonntag1990``: (611.2, 17.62, 243.12)

    Args:
        ta: The air temperature
        util_params: An object of class ~`pyrealm.param_classes.UtilParams`
            giving the settings to be used in conversions.

    Returns:
        Saturated air vapour pressure in kPa.

    Examples:

        >>> # Saturated vapour pressure at 21°C
        >>> round(calc_vp_sat(21), 6)
        2.480904
        >>> from pyrealm.param_classes import UtilParams
        >>> allen = UtilParams(magnus_option='Allen1998')
        >>> round(calc_vp_sat(21, util_params=allen), 6)
        2.487005
        >>> alduchov = UtilParams(magnus_option='Alduchov1996')
        >>> round(calc_vp_sat(21, util_params=alduchov), 6)
        2.481888
    """

    # Magnus equation and conversion to kPa
    cf = util_params.magnus_coef
    vp_sat = cf[0] * np.exp((cf[1] * ta) / (cf[2] + ta)) / 1000

    return vp_sat


def convert_vp_to_vpd(vp, ta, util_params=UtilParams()):
    """Converts vapour pressure to vapour pressure deficit.

    Args:
        vp: The vapour pressure in kPa
        ta: The air temperature in °C
        util_params: An object of class ~`pyrealm.param_classes.UtilParams`
            giving the settings to be used in conversions.
    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> round(convert_vp_to_vpd(1.9, 21), 7)
        0.5809042
        >>> from pyrealm.param_classes import UtilParams
        >>> allen = UtilParams(magnus_option='Allen1998')
        >>> round(convert_vp_to_vpd(1.9, 21, util_params=allen), 7)
        0.5870054
    """
    vp_sat = calc_vp_sat(ta, util_params=util_params)

    return vp_sat - vp


def convert_rh_to_vpd(rh, ta, util_params=UtilParams()):

    """Converts relative humidity to vapour pressure deficit

    Args:
        rh: The relative humidity (proportion in (0,1))
        ta: The air temperature in °C
        util_params: An object of class ~`pyrealm.param_classes.UtilParams`
            giving the settings to be used in conversions.
    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> round(convert_rh_to_vpd(0.7, 21), 7)
        0.7442712
        >>> from pyrealm.param_classes import UtilParams
        >>> allen = UtilParams(magnus_option='Allen1998')
        >>> round(convert_rh_to_vpd(0.7, 21, util_params=allen), 7)
        0.7461016
        >>> convert_rh_to_vpd(71, 21)
        masked
    """

    rh = _constrain_rh(rh)

    vp_sat = calc_vp_sat(ta, util_params=util_params)

    return vp_sat - (rh * vp_sat)


def convert_sh_to_vp(sh, patm, util_params=UtilParams()):
    """Convert specific humidity to vapour pressure

    Args:
        sh: The specific humidity in kg kg-1
        patm: The atmospheric pressure in kPa
        util_params: An object of class ~`pyrealm.param_classes.UtilParams`
            giving the settings to be used in conversions.
    Returns:
        The vapour pressure in kPa
    Examples:
        >>> round(convert_sh_to_vp(0.006, 99.024), 7)
        0.9517451
    """

    return sh * patm / ((1.0 - util_params.mwr) * sh + util_params.mwr)


def convert_sh_to_vpd(sh, ta, patm, util_params=UtilParams()):
    """Convert specific humidity to vapour pressure deficit

    Args:
        sh: The specific humidity in kg kg-1
        ta: The air temperature in °C
        patm: The atmospheric pressure in kPa
        util_params: An object of class ~`pyrealm.param_classes.UtilParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> round(convert_sh_to_vpd(0.006, 21, 99.024), 6)
        1.529159
        >>> from pyrealm.param_classes import UtilParams
        >>> allen = UtilParams(magnus_option='Allen1998')
        >>> round(convert_sh_to_vpd(0.006, 21, 99.024, util_params=allen), 5)
        1.53526
    """

    vp_sat = calc_vp_sat(ta, util_params=util_params)
    vp = convert_sh_to_vp(sh, patm, util_params=util_params)

    return vp_sat - vp



