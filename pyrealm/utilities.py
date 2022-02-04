import numpy as np
import tabulate
from pyrealm.param_classes import HygroParams
from pyrealm.bounds_checker import bounds_checker
# from pandas.core.series import Series

"""
This module provides utility functions shared by modules or providing
extra functions such as conversions for common forcing variable inputs, 
such as hygrometric and radiation conversions.
"""


def check_input_shapes(*args):
    """This helper function validates inputs to check that they are either
    scalars or arrays and then that any arrays of the same shape. It either
    raises an error or returns the common shape or 1 if all arguments are
    scalar.

    Parameters:

        *args: A set of numpy arrays or scalar values

    Returns:

        The common shape of any array inputs or 1 if all inputs are scalar.

    Examples:

        >>> check_input_shapes(np.array([1,2,3]), 5)
        (3,)
        >>> check_input_shapes(4, 5)
        1
        >>> check_input_shapes(np.array([1,2,3]), np.array([1,2]))
        Traceback (most recent call last):
        ...
        ValueError: Inputs contain arrays of different shapes.
    """

    # Collect the shapes of the inputs
    shapes = set()

    # DESIGN NOTES - currently allow:
    #   - scalars,
    #   - 0 dim ndarrays (also scalars but packaged differently)
    #   - 1 dim ndarrays with only a single value

    for val in args:
        if isinstance(val, np.ndarray):
            # Note that 0-dim ndarrays (which are scalars) pass through as do
            # one dimensional arrays with a single value (also a scalar)
            if not(val.ndim == 0 or val.shape == (1,)):
                shapes.add(val.shape)
        # elif isinstance(val, Series):
        #    # Note that 0-dim ndarrays (which are scalars) pass through
        #    if val.ndim > 0:
        #        shapes.add(val.shape)
        elif val is None or isinstance(val, (float, int, np.generic)):
            pass  # No need to track scalars and optional values pass None
        else:
            raise ValueError(f'Unexpected input to check_input_shapes: {type(val)}')

    # shapes can be an empty set (all scalars) or contain one common shape
    # otherwise raise an error
    if len(shapes) > 1:
        raise ValueError('Inputs contain arrays of different shapes.')

    if len(shapes) == 1:
        return shapes.pop()

    return 1


def summarize_attrs(obj, attrs, dp=2, repr_head=True):
    """
    Helper function to create a simple table of attribute mean, min, max and
    nan count from an object for use in summarize function.

    Args:
        obj: An object with attributes to summarize
        attrs: A list of strings of attribute names
        dp: The number of decimal places used in rounding summary stats.
        repr_head: A boolean indicating whether to show the object representation
          before the summary table.

    Returns:
        None
    """
    
    # Create a list to hold variables and summary stats
    ret = []

    for attr in attrs:
        data = getattr(obj, attr)
        
        # Avoid masked arrays - run into problems with edge cases with all NaN 
        if isinstance(data, np.ma.core.MaskedArray):
            data = data.filled(np.nan)
        
        # Add the variable and stats to the list to be displayed
        ret.append([attr,
                    np.round(np.nanmean(data), dp),
                    np.round(np.nanmin(data), dp),
                    np.round(np.nanmax(data), dp),
                    np.count_nonzero(np.isnan(data))])

    hdrs = ['Attr', 'Mean', 'Min', 'Max', 'NaN']

    if repr_head:
        print(obj)

    print(tabulate.tabulate(ret, headers=hdrs))

# Psychrometric conversions to VPD for vapour pressure, specific humidity and
# relative humidity. Using the bigleaf R package as a checking reference from
# which the doctest values are taken


def calc_vp_sat(ta, hygro_params=HygroParams()):

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
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        Saturated air vapour pressure in kPa.

    Examples:

        >>> # Saturated vapour pressure at 21°C
        >>> round(calc_vp_sat(21), 6)
        2.480904
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(calc_vp_sat(21, hygro_params=allen), 6)
        2.487005
        >>> alduchov = HygroParams(magnus_option='Alduchov1996')
        >>> round(calc_vp_sat(21, hygro_params=alduchov), 6)
        2.481888
    """

    # Magnus equation and conversion to kPa
    cf = hygro_params.magnus_coef
    vp_sat = cf[0] * np.exp((cf[1] * ta) / (cf[2] + ta)) / 1000

    return vp_sat


def convert_vp_to_vpd(vp, ta, hygro_params=HygroParams()):
    """Converts vapour pressure to vapour pressure deficit.

    Args:
        vp: The vapour pressure in kPa
        ta: The air temperature in °C
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.
    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> round(convert_vp_to_vpd(1.9, 21), 7)
        0.5809042
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(convert_vp_to_vpd(1.9, 21, hygro_params=allen), 7)
        0.5870054
    """
    vp_sat = calc_vp_sat(ta, hygro_params=hygro_params)

    return vp_sat - vp


def convert_rh_to_vpd(rh, ta, hygro_params=HygroParams()):

    """Converts relative humidity to vapour pressure deficit

    Args:
        rh: The relative humidity (proportion in (0,1))
        ta: The air temperature in °C
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.
    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> round(convert_rh_to_vpd(0.7, 21), 7)
        0.7442712
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(convert_rh_to_vpd(0.7, 21, hygro_params=allen), 7)
        0.7461016
        >>> import sys; sys.stderr = sys.stdout
        >>> round(convert_rh_to_vpd(70, 21), 7)
        pyrealm/bounds_checker.py:104: UserWarning: Variable rh (proportion) contains values outside the expected range (0,1). Check units?
        -171.1823864
    """

    rh = bounds_checker(rh, 0, 1, '[]', 'rh', 'proportion')

    vp_sat = calc_vp_sat(ta, hygro_params=hygro_params)

    return vp_sat - (rh * vp_sat)


def convert_sh_to_vp(sh, patm, hygro_params=HygroParams()):
    """Convert specific humidity to vapour pressure

    Args:
        sh: The specific humidity in kg kg-1
        patm: The atmospheric pressure in kPa
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.
    Returns:
        The vapour pressure in kPa
    Examples:
        >>> round(convert_sh_to_vp(0.006, 99.024), 7)
        0.9517451
    """

    return sh * patm / ((1.0 - hygro_params.mwr) * sh + hygro_params.mwr)


def convert_sh_to_vpd(sh, ta, patm, hygro_params=HygroParams()):
    """Convert specific humidity to vapour pressure deficit

    Args:
        sh: The specific humidity in kg kg-1
        ta: The air temperature in °C
        patm: The atmospheric pressure in kPa
        hygro_params: An object of class ~`pyrealm.param_classes.HygroParams`
            giving the settings to be used in conversions.

    Returns:
        The vapour pressure deficit in kPa

    Examples:
        >>> round(convert_sh_to_vpd(0.006, 21, 99.024), 6)
        1.529159
        >>> from pyrealm.param_classes import HygroParams
        >>> allen = HygroParams(magnus_option='Allen1998')
        >>> round(convert_sh_to_vpd(0.006, 21, 99.024, hygro_params=allen), 5)
        1.53526
    """

    vp_sat = calc_vp_sat(ta, hygro_params=hygro_params)
    vp = convert_sh_to_vp(sh, patm, hygro_params=hygro_params)

    return vp_sat - vp



