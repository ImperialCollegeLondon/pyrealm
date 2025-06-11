"""This module defines a constant class for use with the two leaf, two stream model
:cite:t:`depury:1997a` of assimilation, implemented in the
:mod:`~pyrealm.pmodel.two_leaf` module.
"""  # noqa D210, D415

from dataclasses import dataclass, field

import numpy as np

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class TwoLeafConst(ConstantsClass):
    r"""Constants for the two leaf, two stream model.

    The derived constant for the reflectance of horizontal leaves (:math:`\rho_h`)
    is calculated automatically from the leaf scattering coefficient (:math:`\sigma`),
    following equation A20 of :cite:t:`depury:1997a`.

    .. math::

        \rho_h = \frac{1 - \sqrt{1 - \sigma}}{1 + \sqrt{1 - \sigma}}
    """

    atmospheric_scattering_coef: float = 0.426  # needs citation
    """Scattering coefficient of PAR in the atmosphere (:math:`f_a`, dimensionless)"""
    leaf_scattering_coef: float = 0.15
    r"""Scattering coefficient of PAR by leaves by reflection and transmission
    (:math:`\sigma`, dimensionless, Table 2. of :cite:t:`depury:1997a`)"""
    diffuse_reflectance: float = 0.036  # needs citation
    r"""Canopy reflection coefficient for diffuse PAR,(:math:`\rho_{cd}`,
    dimensionless)."""
    diffuse_extinction_coef: float = 0.719  # needs citation
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
    atmos_transmission_par: float = 0.72
    """The atmospheric transmission coefficient of photosynthetically active radiation,
    used in the calculation of fraction of diffuse light reaching the canopy
    (:math:`a`, Table 5. of :cite:t:`depury:1997a`)."""
    vcmax_lloyd_coef: tuple[float, float] = (0.00963, 2.43)
    """Coefficients of the canopy extinction coefficient (:math:`k_v`) function, taken
    from Figure 10 of :cite:`lloyd:2010a`"""
    jmax25_wullschleger_coef: tuple[float, float] = (29.1, 1.64)
    r"""Coefficients of the empirical relationship between the maximum rate of electron
     transport (:math:`J_{max25}`) and the carboxylation rate (:math:`V_{cmax25}`),
     values taken from :cite:`wullschleger:1993a`."""

    horizontal_leaf_reflectance: float = field(init=False)
    r"""The reflectance coefficient for horizontal leaves (:math:`\rho_h`)."""

    def __post_init__(self) -> None:
        """Set derived constants.

        This requires setattr because the dataclass is frozen.
        """

        object.__setattr__(
            self,
            "horizontal_leaf_reflectance",
            (1 - np.sqrt(1 - self.leaf_scattering_coef))
            / (1 + np.sqrt(1 - self.leaf_scattering_coef)),
        )
