"""The phenology_const module TODO."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class PhenologyConst(ConstantsClass):
    r"""Model constants for the phenology module class.

    This data class defines constants used in the phenology module.
    """

    z: float = 12.227
    r"""The constant :math:`z` (mol C m^{-2} year^{-1}) accounts for the carbon costs of
    building and maintaining leaves and the total below-ground allocation required to
    support the nutrient demand of those leaves. The default is an empirical estimate
    from global data."""

    k: float = 0.5
    """The canopy light extinction coefficient (unitless)"""

    f0_coefficients: tuple[float, float, float] = (0.65, 0.604169, 1.9)
    r"""Coefficients :math:`a,b,c` to calculate :math:`f_0` from the local aridity index
    (AI), where :math:`a` is the maximum value, :math:`b` is the slope of the
    relationship with AI and :math:`c` is the AI value at which the maximum value
    :math:`a` is reached.
    """

    sigma: float = 0.771
    r"""The :math:`\sigma` parameter captures the ability of plants to maintain maximum
    :term:`LAI` throughout the growing season. If a canopy with maximum LAI grows
    instantaneously and is present until the end of the growing season, then the
    potential LAI profile forms a square wave (:math:`\sigma = 1`). However real plants
    require time to deploy a canopy at the start of the growing season, through bud
    formation, budburst and leaf growth. Time is also needed at the end of the season
    for processes such as nutrient resorption and leaf sensescence. These processes
    round off that square wave and this is captured by values of :math:`\sigma < 1`."""

    def calculate_f0(self, aridity_index: NDArray[np.floating]) -> NDArray[np.floating]:
        r"""Calculate the :math:`f_0` parameter.

        The value :math:`f_0` is the ratio of annual total transpiration of annual total
        precipitation. It is calculated from site specific estimates of the
        climatological aridity index, calculated as the long term (typically 20 years)
        total PET over total precipitation (:math:`AI`, unitless) as:


        .. math::

                f_0 = a \exp{\left(-b \left(\frac{AI}{c}\right)^2\right)}

        where :math:`a,b,c` are defined in the
        :attr:`~pyrealm.constants.phenology_const.PhenologyConst.f0_coefficients`
        attribute.
        """

        a, b, c = self.f0_coefficients
        return a * np.exp(-b * np.log(aridity_index / c) ** 2)


@dataclass(frozen=True)
class PhenologyConstNew(ConstantsClass):
    r"""Model constants for the phenology module class.

    This data class defines constants used in the phenology module. The different
    methods for calculating maximum fAPAR and the phenology use different settings, so
    these constants define some method specific settings.

    Two critical values used in calculating maximum fAPAR are:

    * The value :math:`z` (mol C m^{-2} year^{-1}) accounts for the carbon costs of
      building and maintaining leaves and the total below-ground allocation required to
      support the nutrient demand of those leaves. It is used to calculate an
      energy-limited estimate of maximum fAPAR.

    * The value :math:`f_0` (-) accounts for water limitation on annual assimilation and
      hence sets the water-limited estimate of maximum fAPAR.

    Different methods use different default values for these two parameters and they are
    not necessarily fixed: the method of :cite:`cai:2025a` calculates :math:`f_0` as a
    function of the aridity index for a site. These method specific differences in
    parameterisation are explained in the descriptions of the method implementations.
    """

    k: float = 0.5
    """The canopy light extinction coefficient (unitless)"""

    cai_z: float = 12.227
    r"""A value for :math:`z` used by :cite:`cai:2025a`, calculated as an empirical
    estimate from global data."""

    cai_f0_coefficients: tuple[float, float, float] = (0.65, 0.604169, 1.9)
    r"""The method of :cite:`cai:2025a` calculates :math:`f_0` as a function of a site
    specific climatological aridity index (AI). This attribute provides the coefficients
    :math:`a,b,c`, where :math:`a` is the maximum value, :math:`b` is the slope of the
    relationship with AI and :math:`c` is the AI value at which the maximum value
    :math:`a` is reached.
    """

    cai_sigma: float = 0.771
    r"""In the method of :cite:`cai:2025a`, :math:`\sigma`  captures the ability of
    plants to maintain maximum :term:`LAI` throughout the growing season. If a canopy
    with maximum LAI grows instantaneously and is present until the end of the growing
    season, then the potential LAI profile forms a square wave (:math:`\sigma = 1`).
    However real plants require time to deploy a canopy at the start of the growing
    season, through bud formation, budburst and leaf growth. Time is also needed at the
    end of the season for processes such as nutrient resorption and leaf sensescence.
    These processes round off that square wave and this is captured by values of
    :math:`\sigma < 1`."""

    zhou_alpha: float = 1 / 15
    """In the method of :cite:`zhou:2025a`, this is alpha value used with the
    exponential weighted average to set the magnitude of the lagging between realised
    LAI and the steady state values."""

    zhu_f0: float = 0.5
    r"""The value of :math:`f_0` used to estimate water limited fAPAR in the method of
    :cite:`zhu:2026a`."""

    zhu_z: float = 17
    r"""The value of :math:`z` used to estimate energy limited fAPAR in the method of
    :cite:`zhu:2026a`."""

    zhu_budyko: float = 4
    r"""The value of :math:`\bar\omega` used to estimate maximum fAPAR in the method of
    :cite:`zhu:2026a`."""

    zhu_A0_quantile = 0.95
    r"""A quantile of the daily assimilation distribution within a year used to
    calculate the allocation of potential GPP to LAI (:math:`m`), used by
    :cite:`zhu:2026a`."""

    zhu_min_a0 = 1.0e-6
    r"""A lower bound on daily assimilation when calculating leaf area index, used by
    :cite:`zhu:2026a`."""

    zhu_lagcoef = 64.6
    r"""A coefficient used in :cite:`zhu:2026a` to estimate the effect of the
    climatological AET/PET ratio for a site on the lag length used to estimate realised
    LAI from steady state."""

    zhu_lagmax = 365
    r"""An upper bound on the lag length for estimating realised LAI used in
    :cite:`zhu:2026a`."""
