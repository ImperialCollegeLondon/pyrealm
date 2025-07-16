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
