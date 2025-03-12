"""The :mod:`~pyrealm.core.water` submodule contains core functions for calculating the
density and viscosity of water given the air temperature and atmospheric pressure.
"""  # noqa D210, D415

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.utilities import check_input_shapes, evaluate_horner_polynomial


def calc_density_h2o_chen(
    tc: NDArray[np.float64],
    p: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    """Calculate the density of water using Chen et al 2008.

    This function calculates the density of water at a given temperature and pressure
    (kg/m^3) following :cite:t:`chen:2008a`.

    Warning:
        The predictions from this function are numerically unstable around -58°C.

    Args:
        tc: Air temperature (°C)
        p: Atmospheric pressure (Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`,
            providing the polynomial coefficients for the  :cite:t:`chen:2008a`
            equations.

    Returns:
        Water density in kg/m^3

    Raises:
        ValueError: if the inputs have incompatible shapes.

    Examples:
        >>> round(calc_density_h2o_chen(20, 101325), 3)
        np.float64(998.25)
    """

    # Calculate density at 1 atm (kg/m^3):
    po_coef = core_const.chen_po
    po = evaluate_horner_polynomial(tc, po_coef)

    # Calculate bulk modulus at 1 atm (bar):
    ko_coef = core_const.chen_ko
    ko = evaluate_horner_polynomial(tc, ko_coef)

    # Calculate temperature dependent coefficients:
    ca_coef = core_const.chen_ca
    ca = evaluate_horner_polynomial(tc, ca_coef)

    cb_coef = core_const.chen_cb
    cb = evaluate_horner_polynomial(tc, cb_coef)

    # Convert atmospheric pressure to bar (1 bar = 100000 Pa)
    pbar = (1.0e-5) * p

    pw = ko + ca * pbar + cb * pbar**2.0
    pw /= ko + ca * pbar + cb * pbar**2.0 - pbar
    pw *= (1e3) * po
    return pw


def calc_density_h2o_fisher(
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    """Calculate water density.

    Calculates the density of water as a function of temperature and atmospheric
    pressure (kg/m^3), using the Tumlirz Equation and coefficients calculated by
    :cite:t:`Fisher:1975tm`.

    Warning:
        The predictions from this function are unstable around -45°C.

    Args:
        tc: air temperature, °C
        patm: atmospheric pressure, Pa
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`,
            providing the polynomial coefficients for the :cite:t:`Fisher:1975tm`
            equations.

    Returns:
        Water density in kg/m^3.

    Raises:
        ValueError: if the inputs have incompatible shapes.

    Examples:
        >>> round(calc_density_h2o_fisher(20, 101325), 3)
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


def calc_density_h2o(
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
    safe: bool = True,
) -> NDArray[np.float64]:
    """Calculate water density.

    Calculates the density of water as a function of temperature and atmospheric
    pressure (in kg/m^3). This function uses either the method provided by
    :cite:t:`Fisher:1975tm` (:func:`~pyrealm.core.water.calc_density_h2o_fisher`) or
    :cite:t:`chen:2008a` (:func:`~pyrealm.core.water.calc_density_h2o_chen`).

    The constants attribute
    :attr:`~pyrealm.constants.core_const.CoreConst.water_density_method` can be used to
    set which of the ``fisher`` or ``chen`` methods is used.

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
        >>> round(calc_density_h2o(20, 101325), 3)
        np.float64(998.206)
    """

    # Safe guard against instability in functions at low temperature.
    if safe and np.nanmin(tc) < np.array([-30]):
        raise ValueError(
            "Water density calculations below about -30°C are "
            "unstable. See argument safe to calc_density_h2o"
        )

    # Check input shapes, shape not used
    _ = check_input_shapes(tc, patm)

    if core_const.water_density_method == "fisher":
        return calc_density_h2o_fisher(tc, patm, core_const)

    if core_const.water_density_method == "chen":
        return calc_density_h2o_chen(tc, patm, core_const)

    raise ValueError("Unknown method provided to calc_density_h2o")


def calc_viscosity_h2o(
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
    simple: bool = False,
) -> NDArray[np.float64]:
    r"""Calculate the viscosity of water.

    Calculates the viscosity of water (:math:`\eta`) as a function of temperature and
    atmospheric pressure :cite:p:`Huber:2009fy`.

    Args:
        tc: air temperature (°C)
        patm: atmospheric pressure (Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`
        simple: Use the simple formulation.

    Returns:
        A float giving the viscosity of water (mu, Pa s)

    Examples:
        >>> # Density of water at 20 degrees C and standard atmospheric pressure:
        >>> round(calc_viscosity_h2o(20, 101325), 7)
        np.float64(0.0010016)
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    if simple or core_const.simple_viscosity:
        # The reference for this is unknown, but is used in some implementations
        # so is included here to allow intercomparison.
        return np.exp(-3.719 + 580 / ((tc + 273) - 138))

    # Get the density of water, kg/m^3
    rho = calc_density_h2o(tc, patm, core_const=core_const)

    # Calculate dimensionless parameters:
    tbar = (tc + core_const.k_CtoK) / core_const.huber_tk_ast
    rbar = rho / core_const.huber_rho_ast

    # Calculate mu0 (Eq. 11 & Table 2, Huber et al., 2009):
    mu0 = core_const.huber_H_i[0] + core_const.huber_H_i[1] / tbar
    mu0 += core_const.huber_H_i[2] / (tbar * tbar)
    mu0 += core_const.huber_H_i[3] / (tbar * tbar * tbar)
    mu0 = (1e2 * np.sqrt(tbar)) / mu0

    # Calculate mu1 (Eq. 12 & Table 3, Huber et al., 2009):
    ctbar = (1.0 / tbar) - 1.0
    mu1 = 0.0

    # Iterate over the rows of the H_ij core_constants matrix
    for row_idx in np.arange(core_const.huber_H_ij.shape[1]):
        cf1 = ctbar**row_idx
        cf2 = 0.0
        for col_idx in np.arange(core_const.huber_H_ij.shape[0]):
            cf2 += core_const.huber_H_ij[col_idx, row_idx] * (rbar - 1.0) ** col_idx
        mu1 += cf1 * cf2

    mu1 = np.exp(rbar * mu1)

    # Calculate mu_bar (Eq. 2, Huber et al., 2009), assumes mu2 = 1
    mu_bar = mu0 * mu1

    # Calculate mu (Eq. 1, Huber et al., 2009)
    return mu_bar * core_const.huber_mu_ast  # Pa s


def calc_viscosity_h2o_matrix(
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
    simple: bool = False,
) -> NDArray[np.float64]:
    r"""Calculate the viscosity of water.

    Calculates the viscosity of water (:math:`\eta`) as a function of temperature and
    atmospheric pressure :cite:p:`Huber:2009fy`.

    Args:
        tc: air temperature (°C)
        patm: atmospheric pressure (Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`
        simple: Use the simple formulation.

    Returns:
        A float giving the viscosity of water (mu, Pa s)

    Examples:
        >>> # Viscosity of water at 20 degrees C and standard atmospheric pressure:
        >>> round(calc_viscosity_h2o(20, 101325), 7)
        np.float64(0.0010016)
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    if simple or core_const.simple_viscosity:
        # The reference for this is unknown, but is used in some implementations
        # so is included here to allow intercomparison.
        return np.exp(-3.719 + 580 / ((tc + 273) - 138))

    # Get the density of water, kg/m^3
    rho = calc_density_h2o(tc, patm, core_const=core_const)

    # Calculate dimensionless parameters:
    tbar = (tc + core_const.k_CtoK) / core_const.huber_tk_ast
    rbar = rho / core_const.huber_rho_ast

    # Calculate mu0 (Eq. 11 & Table 2, Huber et al., 2009):
    tbar_pow = np.power.outer(tbar, np.arange(0, 4))
    mu0 = (1e2 * np.sqrt(tbar)) / np.sum(
        np.array(core_const.huber_H_i) / tbar_pow, axis=-1
    )

    # Calculate mu1 (Eq. 12 & Table 3, Huber et al., 2009):
    h_array = np.array(core_const.huber_H_ij)
    ctbar = (1.0 / tbar) - 1.0
    row_j, _ = np.indices(h_array.shape)
    mu1 = h_array * np.power.outer(rbar - 1.0, row_j)
    mu1 = np.power.outer(ctbar, np.arange(0, 6)) * np.sum(mu1, axis=(-2))
    mu1 = np.exp(rbar * mu1.sum(axis=-1))

    # Calculate mu_bar (Eq. 2, Huber et al., 2009), assumes mu2 = 1
    mu_bar = mu0 * mu1

    # Calculate mu (Eq. 1, Huber et al., 2009)
    return mu_bar * core_const.huber_mu_ast  # Pa s


def convert_water_mm_to_moles(
    water_mm: NDArray[np.float64],
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
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
        >>> round(convert_water_mm_to_moles(water_mm=1, tc=0, patm=101325), 3)
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
    water_moles: NDArray[np.float64],
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
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
        >>> round(convert_water_moles_to_mm(water_moles=1, tc=0, patm=101325), 3)
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
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
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
        >>> round(calculate_water_molar_volume(0, 101235), 3)
        np.float64(18.015)
    """
    # Calculate density at given temperature and pressure in g/cm3
    water_density = calc_density_h2o(tc=tc, patm=patm, core_const=core_const) / 1000
    # Hence molar volume as mol/cm3 or equivalently mol/mL
    return core_const.k_water_molmass / water_density
