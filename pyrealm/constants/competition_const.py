"""The competition_const module TODO."""
from dataclasses import dataclass

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class C3C4Const(ConstantsClass):
    r"""Model constants for the C3C4Competition class.

    This data class holds statistically estimated coefficients used to calculate the
    fraction of C4 plants based on the relative GPP of C3 and C4 plants for given
    conditions and estimated treecover :cite:p:`lavergne:2020a`.
    """

    # Non-linear regression of fraction C4 plants from proportion GPP advantage
    # of C4 over C3 plants
    adv_to_frac_k = 6.63
    """Coefficient k of C4 fraction estimation"""
    adv_to_frac_q = 0.16
    """Coefficient 1 of C4 fraction estimation"""

    # Conversion parameters to estimate tree cover from  C3 GPP
    gpp_to_tc_a = 15.60
    """Coefficient a of tree cover estimation"""
    gpp_to_tc_b = 1.41
    """Coefficient b of tree cover estimation"""
    gpp_to_tc_c = -7.72
    """Coefficient c of tree cover estimation"""
    c3_forest_closure_gpp = 2.8
    """GPP at which forest canopy closure occurs"""
