"""The :mod:`~pyrealm.core.pressure` submodule contains core functions for calculating
atmospheric presssure. 
"""  # noqa D210, D415

from numpy.typing import NDArray

from pyrealm.constants import PModelConst


def calc_patm(elv: NDArray, const: PModelConst = PModelConst()) -> NDArray:
    r"""Calculate atmospheric pressure from elevation.

    Calculates atmospheric pressure as a function of elevation with reference to the
    standard atmosphere.  The elevation-dependence of atmospheric pressure is computed
    by assuming a linear decrease in temperature with elevation and a mean adiabatic
    lapse rate (Eqn 3, :cite:alp:`BerberanSantos:2009bk`):

    .. math::

        p(z) = p_0 ( 1 - L z / K_0) ^{ G M / (R L) },

    Args:
        elv: Elevation above sea-level (:math:`z`, metres above sea level.)
        const: Instance of :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    PModel Parameters:
        G: gravity constant (:math:`g`, ``k_G``)
        Po: standard atmospheric pressure at sea level (:math:`p_0`, ``k_Po``)
        L: adiabatic temperature lapse rate (:math:`L`, ``k_L``),
        M: molecular weight for dry air (:math:`M`, ``k_Ma``),
        R: universal gas constant (:math:`R`, `k_R``)
        Ko: reference temperature in Kelvin (:math:`K_0`, ``k_To``).

    Returns:
        A numeric value for :math:`p` in Pascals.

    Examples:
        >>> # Standard atmospheric pressure, in Pa, corrected for 1000 m.a.s.l.
        >>> round(calc_patm(1000), 2)
        90241.54
    """

    # Convert elevation to pressure, Pa. This equation uses the base temperature
    # in Kelvins, while other functions use this constant in the PARAM units of
    # °C.

    kto = const.k_To + const.k_CtoK

    return const.k_Po * (1.0 - const.k_L * elv / kto) ** (
        const.k_G * const.k_Ma / (const.k_R * const.k_L)
    )
