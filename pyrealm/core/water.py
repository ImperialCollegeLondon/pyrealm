"""This module provides implementations of functions for the calculation of water
density and viscosity. The methods vary greatly in their complexity also in whether they
correct for the effects of atmospheric pressure. The most precise implementations are
several orders of magnitude slower than the fastest, and differences in prediction are
usually slight.

The various methods are organised into two registry dictionaries that provide a simple
lookup method to select the implementation for use through settings to a
:attr:`~pyrealm.constants.CoreConst` instance. The two decorators
:meth:`register_density_method` and :meth:`register_viscosity_method` are used to add
methods to the respective :data:`DENSITY_METHODS` and :data:`VISCOSITY_METHODS`
registries. In order to maintain a consistent API for calling methods, all methods
within a registry must use the same function signature **even if the method does not use
all of the parameters**: functions that do not correct for atmospheric pressure must
still require that argument.

These registries can be extended by users to provide their own calculations of density
and viscosity.
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Callable
from inspect import signature

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.utilities import check_input_shapes, evaluate_horner_polynomial

DENSITY_METHODS: dict[str, Callable] = {}
"""A registry for functions calculating water density. All registered functions are
expected to have the same signature."""

DENSITY_FUNCTION_SIGNATURE: tuple[tuple[str, str], ...] = (
    ("tc", "NDArray[np.floating]"),
    ("patm", "NDArray[np.floating]"),
    ("core_const", "CoreConst"),
)
"""The expected signature for registered water density functions, as tuples of parameter
name and type annotation."""


def register_density_method(method_name: str) -> Callable:
    """Registration decorator for water density functions.

    Functions decorated with ``register_density_method`` are automatically added to the
    ``DENSITY_METHODS`` registry when imported, using the provided name as a key.

    Args:
        method_name: A short name used as a key for the function in the registry.
    """

    def decorator(function: Callable) -> Callable:
        function_signature = tuple(
            (p.name, p.annotation) for p in signature(function).parameters.values()
        )
        if function_signature != DENSITY_FUNCTION_SIGNATURE:
            raise RuntimeError("Density function does not have expected signature.")

        DENSITY_METHODS[method_name] = function
        return function

    return decorator


@register_density_method("kell")
def calculate_density_h2o_kell(
    tc: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate density of water by Kell's method.

    This function calculates water density as a function of temperature (:math:`T`, °C)
    following Eqn 16. in :cite:`kell:1975a`. This method does not correct for
    atmospheric pressure.

    .. math::

        \rho = \frac{a + bT + cT^2 + dT^3 + eT^4 + fT^5}{1 + gT}

    with coefficients :math:`a,b,c,d,e,f,g` defined in
    :attr:`CoreConst.density_kell<pyrealm.constants.CoreConst.density_kell>`

    Args:
        tc: Temperature in °C
        patm: Atmospheric pressure in Pa
        core_const: An instance of CoreConst providing coefficients

    Examples:
        >>> calculate_density_h2o_kell(np.array([20]), np.array([101325])).round(3)
        array([997.936])
    """
    poly, denom = core_const.density_kell
    return evaluate_horner_polynomial(tc, poly) / (1 + denom * tc)


@register_density_method("jones_harris_eq6")
def calculate_density_h2o_jones_harris_eq6(
    tc: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate density of water following Jones and Harris Eqn 6.

    This function calculates water density as a function of temperature (:math:`T`, °C)
    following Eqn 6. in :cite:`jones:1992a`.

    .. math::

        \rho_{as} = a + bT + cT^2 + dT^3 + eT^4

    with coefficients :math:`a,b,c,d,e` defined in
    :attr:`CoreConst.density_jones_harris_rho<pyrealm.constants.CoreConst.density_jones_harris_rho>`
    This is the density of air-saturated water using the ITS-90 definition of
    temperature and is intended for use in the range 5-40°C. This method does not
    correct for atmospheric pressure.

    Args:
        tc: Temperature in °C
        patm: Atmospheric pressure in Pa
        core_const: An instance of CoreConst providing coefficients

    Examples:
        >>> calculate_density_h2o_jones_harris_eq6(
        ...     np.array([20]), np.array([101325])
        ... ).round(3)
        array([998.201])
    """
    return evaluate_horner_polynomial(tc, core_const.density_jones_harris_rho)


@register_density_method("jones_harris_eq8")
def calculate_density_h2o_jones_harris_eq8(
    tc: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate density of water following Jones and Harris Eqn 8.

    This function calculates water density (:math:`\rho`, kg/m3) as a function of
    temperature (:math:`T`, °C) and pressure (:math:`P`, Pa) following Eqn 8. in
    :cite:`jones:1992a`.

    .. math::

        \rho_{asc} = \rho_{as}( 1+ \kappa_T(P / 1000 - 101.325))

    where :math:`\rho_{as}` is calculated as in
    :meth:`calculate_density_h2o_jones_harris_eq6` and:

    .. math::

        \kappa_T = a + bT + cT^2 + dT^3 + eT^4

    with coefficients :math:`a,b,c,d,e` defined in
    :attr:`CoreConst.density_jones_harris_kappa<pyrealm.constants.CoreConst.density_jones_harris_kappa>`.
    This is the density of air-saturated water using the ITS-90 definition of
    temperature and is intended for use in the range 5-40°C.

    Args:
        tc: Temperature in °C
        patm: Atmospheric pressure in Pa
        core_const: An instance of CoreConst providing coefficients

    Examples:
        >>> calculate_density_h2o_jones_harris_eq8(
        ...     np.array([20]), np.array([101325])
        ... ).round(3)
        array([998.201])
    """
    rho_as = calculate_density_h2o_jones_harris_eq6(
        tc=tc, patm=patm, core_const=core_const
    )
    kappa_t = evaluate_horner_polynomial(tc, core_const.density_jones_harris_kappa)

    return rho_as * (1 + kappa_t * (patm / 1000 - 101.325))


@register_density_method("chen")
def calculate_density_h2o_chen(
    tc: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate the density of water using Chen et al 2008.

    This function calculates the density of water (:math:`\rho`, kg/m^3) as a function
    of temperature (:math:`T`, °C) and atmospheric pressure (:math:`P`, Pa) following
    :cite:t:`chen:2008a`.

    Warning:
        The predictions from this function are numerically unstable around -58°C.

    Args:
        tc: Air temperature (°C)
        patm: Atmospheric pressure (Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`,
            providing the polynomial coefficients for the  :cite:t:`chen:2008a`
            equations.

    Examples:
        >>> calculate_density_h2o_chen(
        ...     np.array([20]), np.array([101325])
        ... ).round(3)
        array([998.25])
    """

    # Calculate density at 1 atm (kg/m^3):
    po = evaluate_horner_polynomial(tc, core_const.chen_po)

    # Calculate bulk modulus at 1 atm (bar):
    ko = evaluate_horner_polynomial(tc, core_const.chen_ko)

    # Calculate temperature dependent coefficients:
    ca = evaluate_horner_polynomial(tc, core_const.chen_ca)
    cb = evaluate_horner_polynomial(tc, core_const.chen_cb)

    # Convert atmospheric pressure to bar (1 bar = 100000 Pa)
    pbar = (1.0e-5) * patm

    pw = ko + ca * pbar + cb * pbar**2.0
    pw /= ko + ca * pbar + cb * pbar**2.0 - pbar
    pw *= (1e3) * po
    return pw


@register_density_method("fisher")
def calculate_density_h2o_fisher(
    tc: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate water density.

    Calculates the density of water (:math:`\rho`, kg/m^3) as a function of temperature
    (:math:`T`, °C) and atmospheric pressure (:math:`P`, Pa), using the Tumlirz Equation
    and coefficients calculated by :cite:t:`Fisher:1975tm`.

    Warning:
        The predictions from this function are unstable around -45°C.

    Args:
        tc: air temperature, °C
        patm: atmospheric pressure, Pa
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`,
            providing the polynomial coefficients for the :cite:t:`Fisher:1975tm`
            equations.

    Examples:
        >>> calculate_density_h2o_fisher(20, 101325).round(3)
        np.float64(998.206)
    """

    # Check input shapes, shape not used
    _ = check_input_shapes(tc, patm)

    # Calculate lambda, (bar cm^3)/g:
    lambda_coef = core_const.fisher_dial_lambda
    lambda_val = evaluate_horner_polynomial(tc, lambda_coef)

    # Calculate po, bar
    po_coef = core_const.fisher_dial_Po
    po_val = evaluate_horner_polynomial(tc, po_coef)

    # Calculate vinf, cm^3/g
    vinf_coef = core_const.fisher_dial_Vinf
    vinf_val = evaluate_horner_polynomial(tc, vinf_coef)

    # Convert pressure to bars (1 bar <- 100000 Pa)
    pbar = 1e-5 * patm

    # Calculate the specific volume (cm^3 g^-1):
    spec_vol = vinf_val + lambda_val / (po_val + pbar)

    # Convert to density in kg/m^3
    rho = 1e3 / spec_vol

    return rho


def calculate_density_h2o(
    tc: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
    safe: bool = True,
) -> NDArray[np.floating]:
    """Calculate water density.

    Calculates the density of water as a function of temperature and atmospheric
    pressure (in kg/m^3), using the method specified in
    :attr:`CoreConst.water_density_method<pyrealm.constants.core_const.CoreConst.water_density_method>`.

    Args:
        tc: air temperature, °C
        patm: atmospheric pressure, Pa
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`
        safe: Prevents the function from estimating density below -30°C, where the
            functions are numerically unstable.

    Returns:
        Water density in kg/m^3.

    Raises:
        ValueError: if ``tc`` contains values below -30°C and ``safe`` is True, or if
            the inputs have incompatible shapes.

    Examples:
        >>> calculate_density_h2o(20, 101325).round(3)
        np.float64(998.206)
    """

    # Safe guard against instability in functions at low temperature.
    if safe and np.nanmin(tc) < np.array([-30]):
        raise ValueError(
            "Water density calculations below about -30°C are "
            "unstable. See argument safe to calculate_density_h2o"
        )

    # Check input shapes, shape not used
    _ = check_input_shapes(tc, patm)

    try:
        func = DENSITY_METHODS[core_const.water_density_method]
    except KeyError:
        raise ValueError(
            f"Unknown density method '{core_const.water_density_method}' "
            "used with calculate_density_h2o"
        )

    return func(tc=tc, patm=patm, core_const=core_const)


VISCOSITY_METHODS: dict[str, Callable] = {}
"""A registry for functions calculating water viscosity. All registered functions are
expected to have the same signature."""

VISCOSITY_FUNCTION_SIGNATURE: tuple[tuple[str, str], ...] = (
    ("tk", "NDArray[np.floating]"),
    ("patm", "NDArray[np.floating]"),
    ("core_const", "CoreConst"),
)
"""The expected signature for registered water viscosity functions, as tuples of parameter
name and type annotation."""


def register_viscosity_method(method_name: str) -> Callable:
    """Registration decorator for water viscosity functions.

    Functions decorated with ``register_viscosity_method`` are automatically added to
    the ``VISCOSITY_METHODS`` registry when imported, using the provided name as a key.

    Args:
        method_name: A short name used as a key for the function in the registry.
    """

    def decorator(function: Callable) -> Callable:
        function_signature = tuple(
            (p.name, p.annotation) for p in signature(function).parameters.values()
        )
        if function_signature != VISCOSITY_FUNCTION_SIGNATURE:
            raise RuntimeError("Viscosity function does not have expected signature.")

        VISCOSITY_METHODS[method_name] = function
        return function

    return decorator


@register_viscosity_method("vogel")
def calculate_viscosity_h2o_vogel(
    tk: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate viscosity of water using the Vogel-Fulcher-Tammann equation.

    This function calculates water viscosity (:math:`\mu`, Pa s) as a function of
    temperature (:math:`T`, K) using the  Vogel-Fulcher-Tammann equation with
    coefficients from :cite:`viswanath:1988a`.

    .. math::

        \mu =  a e^{\frac{b}{T - c}

    with coefficients :math:`a,b,c` defined in
    :attr:`CoreConst.viscosity_vogel<pyrealm.constants.CoreConst.viscosity_vogel>`

    This method does not correct for atmospheric pressure.

    Args:
        tk: air temperature (K)
        patm: Atmospheric pressure in Pa
        core_const: An instance of CoreConst providing coefficients

    Examples:
        >>> calculate_viscosity_h2o_vogel(
        ...     tk=np.array([293.15]), patm=np.array([101325])
        ... ).round(8)
        array([0.00100353])
    """

    A, B, C = core_const.viscosity_vogel
    return A * np.exp(B / (tk - C)) / 1000


@register_viscosity_method("viswanath_natarajan")
def calculate_viscosity_h2o_viswanath_natarajan(
    tk: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate viscosity of water using the Viswanath and Natarajan form.

    This function calculates water viscosity (:math:`\mu`, Pa s) as a function of
    temperature (:math:`T`, K) using the Viswanath and Natarajan form and coefficients
    from Equation 4.18 of :cite:`viswanath:2007a`.

    .. math::

        \log \mu =  a + \frac{b}{c - T}

    with coefficients :math:`a,b,c` defined in
    :attr:`CoreConst.viscosity_viswanath_natarajan<pyrealm.constants.CoreConst.viscosity_viswanath_natarajan>`

    This method does not correct for atmospheric pressure.

    Args:
        tk: air temperature (K)
        patm: Atmospheric pressure in Pa
        core_const: An instance of CoreConst providing coefficients

    Examples:
        >>> calculate_viscosity_h2o_viswanath_natarajan(
        ...     tk=np.array([293.15]), patm=np.array([101325])
        ... ).round(8)
        array([0.00100576])
    """

    A, B, C = core_const.viscosity_viswanath_natarajan
    return 10 ** (A + B / (C - tk))


@register_viscosity_method("girifalco")
def calculate_viscosity_h2o_girifalco(
    tk: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate viscosity of water using the Girifalco form.

    This function calculates water viscosity (:math:`\mu`, Pa s) as a function of
    temperature (:math:`T`, K) using the Girifalco form and coefficients from Equation
    4.27 of :cite:`viswanath:2007a`.

    .. math::

        \log \mu =  a + \frac{b}{T} + \frac{c}{T^2}

    with coefficients :math:`a,b,c` defined in
    :attr:`CoreConst.viscosity_girifalco<pyrealm.constants.CoreConst.viscosity_girifalco>`

    This method does not correct for atmospheric pressure.

    Args:
        tk: air temperature (K)
        patm: Atmospheric pressure in Pa
        core_const: An instance of CoreConst providing coefficients

    Examples:
        >>> calculate_viscosity_h2o_girifalco(
        ...    tk=np.array([293.15]), patm=np.array([101325])
        ... ).round(8)
        array([0.00100742])
    """

    A, B, C = core_const.viscosity_girifalco
    return 10 ** (A + B / tk + C / tk**2) / 1000


@register_viscosity_method("reid")
def calculate_viscosity_h2o_reid(
    tk: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate viscosity of water using the Reid form.

    This function calculates water viscosity (:math:`\mu`, Pa s) as a function of
    temperature (:math:`T`, K) using a polynomial form from Equation 4.39 of
    :cite:`viswanath:2007a` and coefficients taken from Reid et al.

    .. math::

        \log \mu =  a + \frac{b}{T} + cT + dT^2

    with coefficients :math:`a,b,c,d` defined in
    :attr:`CoreConst.viscosity_reid<pyrealm.constants.CoreConst.viscosity_reid>`

    This method does not correct for atmospheric pressure.

    Args:
        tk: air temperature (K)
        patm: Atmospheric pressure in Pa
        core_const: An instance of CoreConst providing coefficients

    Examples:
        >>> calculate_viscosity_h2o_reid(
        ...    tk=np.array([293.15]), patm=np.array([101325])
        ... ).round(8)
        array([0.00101766])
    """

    A, B, C, D = core_const.viscosity_reid
    return np.exp(A + B / tk + C * tk + D * tk**2) / 1000


@register_viscosity_method("daubert_danner")
def calculate_viscosity_h2o_daubert_danner(
    tk: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate viscosity of water using the Daubert Danner form.

    This function calculates water viscosity (:math:`\mu`, Pa s) as a function of
    temperature (:math:`T`, K) using the Daubert and Danner form from Equation 4.41 of
    :cite:`viswanath:2007a`.

    .. math::

        \mu =  \exp \left( a + \frac{b}{T} + c \ln T  + dT^10 \right)

    with coefficients :math:`a,b,c,d` defined in
    :attr:`CoreConst.viscosity_daubert_danner<pyrealm.constants.CoreConst.viscosity_daubert_danner>`

    This method does not correct for atmospheric pressure.

    Args:
        tk: air temperature (K)
        patm: Atmospheric pressure in Pa
        core_const: An instance of CoreConst providing coefficients

    Examples:
        >>> calculate_viscosity_h2o_daubert_danner(
        ...    tk=np.array([293.15]), patm=np.array([101325])
        ... ).round(8)
        array([0.00103321])
    """

    A, B, C, D = core_const.viscosity_daubert_danner
    return np.exp(A + B / tk + C * np.log(tk) + D * tk**10)


@register_viscosity_method("huber")
def calculate_viscosity_h2o_huber(
    tk: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate the viscosity of water.

    Calculates the viscosity of water (:math:`\mu`, Pa s) as a function of temperature
    and atmospheric pressure :cite:p:`Huber:2009fy`.

    Args:
        tk: air temperature (K)
        patm: atmospheric pressure (Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`
        simple: Use the simple formulation.

    Examples:
        >>> # Density of water at 20 degrees C and standard atmospheric pressure:
        >>> calculate_viscosity_h2o_huber(
        ...    tk=np.array([293.15]), patm=np.array([101325])
        ... ).round(8)
        array([0.0010016])
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tk, patm)

    # Get the density of water, kg/m^3
    rho = calculate_density_h2o(tk - core_const.k_CtoK, patm, core_const=core_const)

    # Calculate dimensionless parameters:
    tbar = tk / core_const.huber_tk_ast
    rbar = rho / core_const.huber_rho_ast

    # Calculate mu0 (Eq. 11 & Table 2, Huber et al., 2009):
    mu0 = core_const.huber_H_i[0] + core_const.huber_H_i[1] / tbar
    mu0 += core_const.huber_H_i[2] / (tbar * tbar)
    mu0 += core_const.huber_H_i[3] / (tbar * tbar * tbar)
    mu0 = (1e2 * np.sqrt(tbar)) / mu0

    # Calculate mu1 (Eq. 12 & Table 3, Huber et al., 2009):
    ctbar = (1.0 / tbar) - 1.0
    mu1 = np.zeros_like(rbar)

    # Iterate over the rows of the H_ij core_constants matrix
    for row_idx in np.arange(core_const.huber_H_ij.shape[1]):
        cf1 = ctbar**row_idx
        cf2 = np.zeros_like(rbar)
        for col_idx in np.arange(core_const.huber_H_ij.shape[0]):
            cf2 += core_const.huber_H_ij[col_idx, row_idx] * (rbar - 1.0) ** col_idx
        mu1 += cf1 * cf2

    mu1 = np.exp(rbar * mu1)

    # Calculate mu_bar (Eq. 2, Huber et al., 2009), assumes mu2 = 1
    mu_bar = mu0 * mu1

    # Calculate mu (Eq. 1, Huber et al., 2009)
    return mu_bar * core_const.huber_mu_ast  # Pa s


def calculate_viscosity_h2o(
    tk: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    r"""Calculate the viscosity of water.

    Calculates the viscosity of water (:math:`\mu`, Pa s) as a function of temperature
    (:math:`T`, K) and atmospheric pressure (:math:`P`, Pa),  using the method specified
    in
    :attr:`CoreConst.water_viscosity_method<pyrealm.constants.core_const.CoreConst.water_viscosity_method>`.

    Args:
        tk: air temperature (K)
        patm: atmospheric pressure (Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`

    Examples:
        >>> # Density of water at 20 °C and standard atmospheric pressure:
        >>> calculate_viscosity_h2o(293.15, 101325).round(7)
        np.float64(0.0010016)
    """

    # Check input shapes, shape not used
    _ = check_input_shapes(tk, patm)

    try:
        func = VISCOSITY_METHODS[core_const.water_viscosity_method]
    except KeyError:
        raise ValueError(
            f"Unknown viscosity method '{core_const.water_viscosity_method}' "
            "used with calculate_viscosity_h2o"
        )

    return func(tk=tk, patm=patm, core_const=core_const)


def convert_water_mm_to_moles(
    water_mm: NDArray[np.floating],
    tc: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    """Convert water in mm per square meter to moles.

    This function converts water volumes expressed as mm per m2 into a number of moles
    of water. It accounts for the changing density of water with temperature and
    pressure.

    Args:
        water_mm: Water volume in mm per square meter
        tc: air temperature (°C)
        patm: atmospheric pressure (Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`

    Returns:
        Moles of water (-)

    Examples:
        >>> # At 0°C and 101325 Pa, one mole of water is ~18 g (18 cm3, 0.018 mm m-2).
        >>> # So, 1 mm m2 = 1 / 0.018 = ~55 moles.
        >>> convert_water_mm_to_moles(water_mm=1, tc=0, patm=101325).round(3)
        np.float64(55.508)
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(water_mm, tc, patm)

    # 1 mm per square meter is 1 litre, so convert to mL and then to moles
    return (
        water_mm
        * 1000
        / calculate_water_molar_volume(tc=tc, patm=patm, core_const=core_const)
    )


def convert_water_moles_to_mm(
    water_moles: NDArray[np.floating],
    tc: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    """Convert water in moles to mm per square meter.

    This function converts water volumes expressed as moles into mm per m2. It accounts
    for the changing density of water with temperature and pressure.

    Args:
        water_moles: Water volume in moles
        tc: air temperature (°C)
        patm: atmospheric pressure (Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`

    Returns:
        Water volume in mm per m2

    Examples:
        >>> # At 0°C and 101325 Pa, one mole of water is ~18 g (18 cm3, 0.018 mm m-2).
        >>> # So, 1 mol = 0.018 mm
        >>> convert_water_moles_to_mm(water_moles=1, tc=0, patm=101325).round(3)
        np.float64(0.018)
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(water_moles, tc, patm)

    # 1 mm per square meter is 1 litre, so convert to mL and then to moles
    return (
        water_moles
        * calculate_water_molar_volume(tc=tc, patm=patm, core_const=core_const)
    ) / 1000


def calculate_water_molar_volume(
    tc: NDArray[np.floating],
    patm: NDArray[np.floating],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.floating]:
    """Calculate the volume of a mole of water at a given temperature and pressure.

    Args:
        tc: air temperature (°C)
        patm: atmospheric pressure (Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`

    Returns:
        Water molar volume in mol cm-3, or equivalently mol/mL

    Examples:
        >>> # A mole of water at standard temperature and pressure occupies ~18 cm3.
        >>> calculate_water_molar_volume(0, 101235).round(3)
        np.float64(18.015)
    """
    # Calculate density at given temperature and pressure in g/cm3
    water_density = (
        calculate_density_h2o(tc=tc, patm=patm, core_const=core_const) / 1000
    )
    # Hence molar volume as mol/cm3 or equivalently mol/mL
    return core_const.k_water_molmass / water_density
