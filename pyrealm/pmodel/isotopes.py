"""The :mod:`~pyrealm.pmodel.isotopes` submodule provides the
:class:`~pyrealm.pmodel.isotopes.CalcCarbonIsotopes` class, which is used to calculate
isotopic discrimination within the PModel
"""  # noqa D210, D415

from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import IsotopesConst
from pyrealm.core.utilities import check_input_shapes, summarize_attrs
from pyrealm.pmodel.pmodel import PModel


class CalcCarbonIsotopes:
    r"""Calculate :math:`\ce{CO2}` isotopic discrimination.

    This class estimates the fractionation of atmospheric CO2 by photosynthetic
    pathways to calculate the isotopic compositions and discrimination given the
    predicted optimal chi from a :class:`~pyrealm.pmodel.pmodel.PModel` instance.

    Discrimination against carbon 13 (:math:`\Delta\ce{^{13}C}`)  is calculated
    using C3 and C4 pathways specific methods, and then discrimination against
    carbon 14 is estimated as :math:`\Delta\ce{^{14}C} \approx 2 \times
    \Delta\ce{^{13}C}` :cite:p:`graven:2020a`. For C3 plants,
    :math:`\Delta\ce{^{13}C}` is calculated both including and excluding
    photorespiration, but these are assumed to be equal for C4 plants. The class
    also reports the isotopic composition of leaves and wood.

    Args:
        pmodel: A :class:`~pyrealm.pmodel.pmodel.PModel` instance providing the
            photosynthetic pathway and estimated optimal chi.
        d13CO2: Atmospheric isotopic ratio for Carbon 13
            (:math:`\delta\ce{^{13}C}`, permil).
        D14CO2: Atmospheric isotopic ratio for Carbon 14
            (:math:`\Delta\ce{^{14}C}`, permil).
        isotopes_const: An instance
            of :class:`~pyrealm.constants.isotope_const.IsotopesConst`, parameterizing
            the calculations.
    """

    def __init__(
        self,
        pmodel: PModel,
        D14CO2: NDArray[np.float64],
        d13CO2: NDArray[np.float64],
        isotopes_const: IsotopesConst = IsotopesConst(),
    ):
        # Check inputs are congruent
        _ = check_input_shapes(pmodel.env.tc, d13CO2, D14CO2)

        self.isotopes_const: IsotopesConst = isotopes_const
        """The IsotopesParams instance used to calculate estimates."""
        self.shape: tuple = pmodel.shape
        """Records the common numpy array shape of array inputs."""
        self.c4: bool = pmodel.c4
        """Indicates if estimates calculated for C3 or C4 photosynthesis."""

        # Attributes defined by methods below
        self.Delta13C_simple: NDArray[np.float64]
        r"""Discrimination against carbon 13 (:math:`\Delta\ce{^{13}C}`, permil)
        excluding photorespiration."""
        self.Delta14C: NDArray[np.float64]
        r"""Discrimination against carbon 13 (:math:`\Delta\ce{^{13}C}`, permil)
        including photorespiration."""
        self.Delta13C: NDArray[np.float64]
        r"""Discrimination against carbon 14 (:math:`\Delta\ce{^{14}C}`, permil)
        including photorespiration."""
        self.d13C_leaf: NDArray[np.float64]
        r"""Isotopic ratio of carbon 13 in leaves
        (:math:`\delta\ce{^{13}C}`, permil)."""
        self.d14C_leaf: NDArray[np.float64]
        r"""Isotopic ratio of carbon 14 in leaves
        (:math:`\delta\ce{^{14}C}`, permil)."""
        self.d13C_wood: NDArray[np.float64]
        r"""Isotopic ratio of carbon 13 in wood (:math:`\delta\ce{^{13}C}`, permil),
        given a parameterized post-photosynthetic fractionation."""

        # Could store pmodel, d13CO2, D14CO2 in instance, but really not needed
        # so try and keep this class simple with a minimum of attributes.
        # TODO: map methods for delta13C to C3 and C4.

        if self.c4:
            self.calc_c4_discrimination(pmodel)
        else:
            self.calc_c3_discrimination(pmodel)

        # 14C discrimination is twice the 13C discrimination (Graven et al. 2020)
        self.Delta14C = self.Delta13C * 2

        # Isotopic composition of leaf
        self.d13C_leaf = (d13CO2 - self.Delta13C) / (1 + self.Delta13C / 1000)
        self.d14C_leaf = (D14CO2 - self.Delta14C) / (1 + self.Delta14C / 1000)

        # Isotopic composition of wood considering post-photosynthetic fractionation:
        self.d13C_wood = self.d13C_leaf + self.isotopes_const.frank_postfrac

    def __repr__(self) -> str:
        """Generates a string representation of a CalcCarbonIsotopes instance."""
        return f"CalcCarbonIsotopes(shape={self.shape}, method={self.c4})"

    def calc_c4_discrimination(self, pmodel: PModel) -> None:
        r"""Calculate C4 isotopic discrimination.

        In this method, :math:`\delta\ce{^{13}C}` is calculated from optimal
        :math:`\chi` using an empirical relationship estimated by
        :cite:p:`lavergne:2022a`.

        Examples:
            >>> import numpy as np
            >>> from pyrealm.pmodel.pmodel import PModel
            >>> from pyrealm.pmodel import PModelEnvironment
            >>> from pyrealm.constants import PModelConst
            >>> pmodel_const = PModelConst(beta_cost_ratio_c4=35)
            >>> env = PModelEnvironment(
            ...     tc=np.array([20]),
            ...     patm=np.array([101325]),
            ...     co2=np.array([400]),
            ...     vpd=np.array([1000]),
            ...     fapar=np.array([1]),
            ...     ppfd=np.array([800]),
            ...     pmodel_const=pmodel_const
            ... )
            >>> mod_c4 = PModel(env, method_optchi='c4_no_gamma')
            >>> mod_c4_delta = CalcCarbonIsotopes(mod_c4, d13CO2= -8.4, D14CO2 = 19.2)
            >>> mod_c4_delta.Delta13C.round(4)
            array([5.6636])
            >>> mod_c4_delta.d13C_leaf.round(4)
            array([-13.9844])
        """

        # Equation from C3/C4 paper
        self.Delta13C_simple = (
            self.isotopes_const.lavergne_delta13_a
            + self.isotopes_const.lavergne_delta13_b * pmodel.optchi.chi
        )
        self.Delta13C = self.Delta13C_simple

    def calc_c4_discrimination_vonC(self, pmodel: PModel) -> None:
        r"""Calculate C4 isotopic discrimination.

        In this method, :math:`\delta\ce{^{13}C}` is calculated from optimal
        :math:`\chi` following Equation 1 in :cite:p:`voncaemmerer:2014a`.

        This method is not yet reachable - it needs a method selection argument to
        switch approaches and check C4 methods are used with C4 pmodels. The method is
        preserving experimental code provided by Alienor Lavergne. A temperature
        sensitive correction term is provided in commented code but not used.

        Examples:
            >>> import numpy as np
            >>> from pyrealm.pmodel.pmodel import PModel
            >>> from pyrealm.pmodel import PModelEnvironment
            >>> from pyrealm.constants import PModelConst
            >>> pmodel_const = PModelConst(beta_cost_ratio_c4=35)
            >>> env = PModelEnvironment(
            ...     tc=np.array([20]),
            ...     patm=np.array([101325]),
            ...     co2=np.array([400]),
            ...     vpd=np.array([1000]),
            ...     fapar=np.array([1]),
            ...     ppfd=np.array([800]),
            ...     pmodel_const=pmodel_const
            ... )
            >>> mod_c4 = PModel(env, method_optchi='c4_no_gamma')
            >>> mod_c4_delta = CalcCarbonIsotopes(mod_c4, d13CO2= -8.4, D14CO2 = 19.2)
            >>> # mod_c4_delta.Delta13C.round(4)
            >>> # array([5.2753])
            >>> # mod_c4_delta.d13C_leaf.round(4)
            >>> # array([-13.6036])
        """

        warn("This method is experimental code from Alienor Lavergne")

        # Equation A5 from von Caemmerer et al. (2014)
        # b4 = (-9.483 * 1000) / (273 + self.tc) + 23.89 + 2.2
        # b4 = self.const.vonCaemmerer_b4

        # 13C discrimination (‰): von Caemmerer et al. (2014) Eq. 1
        self.Delta13C_simple = (
            self.isotopes_const.farquhar_a
            + (
                self.isotopes_const.vonCaemmerer_b4
                + (self.isotopes_const.farquhar_b - self.isotopes_const.vonCaemmerer_s)
                * self.isotopes_const.vonCaemmerer_phi
                - self.isotopes_const.farquhar_a
            )
            * pmodel.optchi.chi
        )

        self.Delta13C = self.Delta13C_simple

    def calc_c3_discrimination(self, pmodel: PModel) -> None:
        r"""Calculate C3 isotopic discrimination.

        This method calculates the isotopic discrimination for
        :math:`\Delta\ce{^{13}C}` both with and without the photorespiratory
        effect following :cite:p:`farquhar:1982a`.

        Examples:
            >>> import numpy as np
            >>> from pyrealm.pmodel import PModelEnvironment
            >>> from pyrealm.pmodel.pmodel import PModel
            >>> env = PModelEnvironment(
            ...              tc=np.array([20]), patm=np.array([101325]),
            ...              co2=np.array([400]), vpd=np.array([1000]),
            ...              fapar=np.array([1]), ppfd=np.array([800]),
            ...              theta=np.array([0.4])
            ... )
            >>> mod_c3 = PModel(env, method_optchi='lavergne20_c3')
            >>> mod_c3_delta = CalcCarbonIsotopes(mod_c3, d13CO2= -8.4, D14CO2 = 19.2)
            >>> mod_c3_delta.Delta13C.round(4)
            array([20.4056])
            >>> mod_c3_delta.d13C_leaf.round(4)
            array([-28.2296])
        """

        # 13C discrimination (permil): Farquhar et al. (1982)
        # Simple
        self.Delta13C_simple = (
            self.isotopes_const.farquhar_a
            + (self.isotopes_const.farquhar_b - self.isotopes_const.farquhar_a)
            * pmodel.optchi.chi
        )

        # with photorespiratory effect:
        self.Delta13C = (
            self.isotopes_const.farquhar_a
            + (self.isotopes_const.farquhar_b2 - self.isotopes_const.farquhar_a)
            * pmodel.optchi.chi
            - self.isotopes_const.farquhar_f * pmodel.env.gammastar / pmodel.env.ca
        )

    def summarize(self, dp: int = 2) -> None:
        """Print summary of values estimated in CalcCarbonIsotopes.

        Prints a summary of the variables calculated within an instance of
        CalcCarbonIsotopes including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = (
            ("Delta13C_simple", "permil"),  # ‰
            ("Delta13C", "permil"),
            ("Delta14C", "permil"),
            ("d13C_leaf", "permil"),
            ("d14C_leaf", "permil"),
            ("d13C_wood", "permil"),
        )

        summarize_attrs(self, attrs, dp=dp)
