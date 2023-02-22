"""The tmodel_const module TODO."""
from dataclasses import dataclass

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class TModelTraits(ConstantsClass):
    r"""Trait data settings for a TTree instance.

    This data class provides the value of the key traits used in the T model. The
    default values are taken from Table 1 of :cite:t:`Li:2014bc`. Note that the foliage
    maintenance respiration fraction is not named in the T Model description, but has
    been included as a modifiable trait in this implementation. The traits are shown
    below with mathematical notation, default value and units shown in brackets:
    """

    a_hd: float = 116.0
    """Initial slope of height-diameter relationship (:math:`a`, 116.0, -)"""
    ca_ratio: float = 390.43
    """Initial ratio of crown area to stem cross-sectional area
    (:math:`c`, 390.43, -)"""
    h_max: float = 25.33  # H_m, Maximum tree height (m)
    """Maximum tree height (:math:`H_m`, 25.33, m)"""
    rho_s: float = 200.0  # rho_s, Sapwood density (kgCm−3)
    r"""Sapwood density (:math:`\rho_s`, 200.0, kg Cm-3)"""
    lai: float = 1.8  # L, Leaf area index within the crown (–)
    """Leaf area index within the crown (:math:`L`, 1.8, -)"""
    sla: float = 14.0  # sigma, Specific leaf area (m2 kg−1C)
    r"""Specific leaf area (:math:`\sigma`, 14.0, m2 kg-1 C)"""
    tau_f: float = 4.0  # tau_f, Foliage turnover time (years)
    r"""Foliage turnover time (:math:`\tau_f`, 4.0, years)"""
    tau_r: float = 1.04  # tau_r, Fine-root turnover time (years)
    """Fine-root turnover time (:math:`\tau_r`, 1.04, years)"""
    par_ext: float = 0.5  # k, PAR extinction coefficient (–)
    """PAR extinction coefficient (:math:`k`, 0.5, -)"""
    yld: float = 0.17  # y, Yield_factor (-)
    """Yield_factor (:math:`y`, 0.17, -)"""
    zeta: float = 0.17  # zeta, Ratio of fine-root mass to foliage area (kgCm−2)
    r"""Ratio of fine-root mass to foliage area (:math:`\zeta`, 0.17, kg C m-2)"""
    resp_r: float = 0.913  # r_r, Fine-root specific respiration rate (year−1)
    """Fine-root specific respiration rate (:math:`r_r`, 0.913, year-1)"""
    resp_s: float = 0.044  # r_s, Sapwood-specific respiration rate (year−1)
    """Sapwood-specific respiration rate (:math:`r_s`, 0.044, year-1)"""
    resp_f: float = 0.1  # --- , Foliage maintenance respiration fraction (-)
    """Foliage maintenance respiration fraction (:math:`r_f`,  0.1, -)"""

    # TODO: include range + se, or make this another class TraitDistrib
    #       that can yield a Traits instance drawing from that distribution
