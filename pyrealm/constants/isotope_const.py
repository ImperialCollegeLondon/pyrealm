"""The isotope_const module TODO."""
from dataclasses import dataclass

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class IsotopesParams(ConstantsClass):
    """Settings for calculate carbon isotope discrimination.

    This data class provides values for underlying constants used in the calculation of
    carbon isotope discrimination from P Model instances.

    * For C4 plants, the coefficients for calculating isotopic discrimination are taken
      from :cite:`lavergne:2022a`. The class also provides an alternative, but currently
      implemented, parameterisation using coefficients taken from
      :cite:`voncaemmerer:2014a`.
    * For Cw plants, the coefficients for calculating isotopic discrimination are taken
      from :cite:`farquhar:1982a`.
    * Post-photosynthetic fractionation values are also provided between leaf organic
      matter and alpha-cellulose (:cite:`frank:2015a`) and bulk wood
      (:cite:`badeck:2005a`).

    """

    lavergne_delta13_a = 13.95
    """Intercept for isotopic discrimination for C4 plants"""
    lavergne_delta13_b = -17.04
    """Slope for C4 isotopic discrimination for C4 plants"""

    # Farquhar et al. (1982)
    farquhar_a: float = 4.4
    """Intercept for isotopic discrimination for C3 plants"""
    farquhar_b: float = 29
    """Coefficient for simple isotopic discrimination for C3 plants"""
    farquhar_b2: float = 28
    """Coefficient for isotopic discrimination with photorespiration for C3 plants"""
    farquhar_f: float = 12
    """Coefficient for isotopic discrimination with photorespiration for C3 plants"""

    # vonCaemmerer et al. (2014)
    vonCaemmerer_b4: float = -7.4
    """Alternative coefficient for isotopic discrimination for C4 plants"""
    vonCaemmerer_s: float = 1.8
    """Alternative coefficient for isotopic discrimination for C4 plants"""
    vonCaemmerer_phi: float = 0.5
    """Alternative coefficient for isotopic discrimination for C4 plants"""

    frank_postfrac: float = 2.1
    """Post-photosynthetic fractionation between leaf organic matter and
    alpha-cellulose (permil)"""

    badeck_postfrac: float = 1.9
    """Post-photosynthetic fractionation between leaf organic matter and bulk wood
    (permil)"""
