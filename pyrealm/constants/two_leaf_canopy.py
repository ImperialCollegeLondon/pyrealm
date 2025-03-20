"""To do."""

from dataclasses import dataclass

import numpy as np

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class TwoLeafConst(ConstantsClass):
    """Constants for the two leaf, two stream model."""

    atmospheric_scattering_coefficient: float = 0.426  # needs citation
    """Scattering coefficient of PAR in the atmosphere (:math:`f_a`, dimensionless)"""
    leaf_scattering_coefficient: float = 0.15
    """Scattering coefficient of PAR by leaves by reflection and transmission
    (:math:`\sigma`, dimensionless, Table 2. of :cite:t:`depury:1997a`)"""
    canopy_reflection_coefficient: float = 0.036  # needs citation
    r"""Canopy reflection coefficient for diffuse PAR,(:math:`\rho_{cd}`,
    dimensionless)."""
    diffuse_extinction_coefficient: float = 0.719  # needs citation
    """Diffuse and scattered diffuse PAR extinction coefficient (:math:`k_d'`,
    dimensionless)."""
    solar_obscurity_angle: float = np.pi / 180  # 1 degree in rads, needs a citation
    r"""Solar obscurity angle (:math:`\beta_{ob}` radians)"""
    direct_beam_extinction_numerator: float = 0.5
    r"""Numerator of the extinction beam coefficient calculation (:math:`n`) for direct
     light."""
    scattered_beam_extinction_numerator: float = 0.46
    r"""Numerator of the extinction beam coefficient calculation (:math:`n`) for
     scattered light."""
    leaf_diffusion_factor: float = 0.72
    """Leaf derived factor used in calculation of fraction of diffuse light (:math:`a`,
    Table 5. of :cite:t:`depury:1997a`)."""
    vcmax_lloyd_coef: tuple[float, float] = (0.00963, 2.43)
    """Coefficients of the canopy extinction coefficient (:math:`k_v`) function, taken
    from Figure 10 of :cite:`lloyd:2010a`"""
