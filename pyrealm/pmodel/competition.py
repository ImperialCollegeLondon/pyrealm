"""The :mod:`~pyrealm.pmodel.competition` submodule provides the
:mod:`~pyrealm.pmodel.competition.C3C4Competition` class, which is used to estimate the
expected fraction of C4 plants given the relative photosynthetic advantages of the two
pathways in locations.
"""  # noqa D210, D415

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import C3C4Const
from pyrealm.utilities import check_input_shapes, summarize_attrs


def convert_gpp_advantage_to_c4_fraction(
    gpp_adv_c4: NDArray, treecover: NDArray, const: C3C4Const = C3C4Const()
) -> NDArray:
    r"""Convert C4 GPP advantage to C4 fraction.

    This function calculates an initial estimate of the fraction of C4 plants based on
    the proportional GPP advantage from C4 photosynthesis. The proportion GPP advantage
    :math:`A_4` is converted to an expected fraction of C4 :math:`F_4` plants using a
    logistic equation of :math:`A_4`, where :math:`A_4` is first modulated by percentage
    tree cover (TC):

    .. math::
        :nowrap:

        \[
            \begin{align*}
                A_4^\prime &= \frac{A_4}{e^ {1 / 1 + \text{TC}}} \\
                F_4 &= \frac{1}{1 + e^{k A_4^\prime} - q}
            \end{align*}
        \]

    The parameters are set in the ``params`` instance and are the slope of the equation
    (:math:`k`, :attr:`~pyrealm.constants.competition_const.C3C4Const.adv_to_frac_k`)
    and :math:`A_4` value at the midpoint of the curve
    (:math:`q`, :attr:`~pyrealm.constants.competition_const.C3C4Const.adv_to_frac_q`).

    Args:
        gpp_adv_c4: The proportional GPP advantage of C4 photosynthesis.
        treecover: The proportion tree cover.

    Returns:
        The estimated fraction of C4 plants given the estimated C4 GPP advantage and
        tree cover.
    """

    frac_c4 = 1.0 / (
        1.0
        + np.exp(
            -const.adv_to_frac_k
            * ((gpp_adv_c4 / np.exp(1 / (1 + treecover))) - const.adv_to_frac_q)
        )
    )

    return frac_c4


def calculate_tree_proportion(
    gppc3: NDArray, const: C3C4Const = C3C4Const()
) -> NDArray:
    r"""Calculate the proportion of GPP from C3 trees.

    This function estimates the proportion of C3 trees in the community, which can then
    be used to penalise the fraction of C4 plants due to shading of C4 plants by canopy
    closure, even when C4 photosynthesis is advantagious. The estimated tree cover
    function is:

        .. math::
            :nowrap:

                \[
                    TC(\text{GPP}_{C3}) = a \cdot \text{GPP}_{C3} ^ b - c
                \]

    with parameters set in the `const` instance (:math:`a`,
    :attr:`~pyrealm.constants.competition_const.C3C4Const.gpp_to_tc_a`; :math:`b`,
    :attr:`~pyrealm.constants.competition_const.C3C4Const.gpp_to_tc_b`; :math:`c`,
    :attr:`~pyrealm.constants.competition_const.C3C4Const.gpp_to_tc_c`). The proportion
    of GPP from C3 trees (:math:`h`) is then estimated using the predicted tree cover in
    locations relative to a threshold GPP value (:math:`\text{GPP}_{CLO}`,
    :attr:`~pyrealm.constants.competition_const.C3C4Const.c3_forest_closure_gpp`) above
    which canopy closure occurs. The value of :math:`h` is clamped in :math:`[0, 1]`:

        .. math::
            :nowrap:

                \[
                    h = \max\left(0, \min\left(
                        \frac{TC(\text{GPP}_{C3})}{TC(\text{GPP}_{CLO})}\right),
                        1 \right)
                \]

    Args:
        gppc3: The estimated GPP for C3 plants. The input values here must be
          expressed  as **kilograms** per metre squared per year (kg m-2 yr-1).

    Returns:
        The estimated proportion of GPP resulting from C3 trees.
    """

    prop_trees = (
        const.gpp_to_tc_a * np.power(gppc3, const.gpp_to_tc_b) + const.gpp_to_tc_c
    ) / (
        const.gpp_to_tc_a * np.power(const.c3_forest_closure_gpp, const.gpp_to_tc_b)
        + const.gpp_to_tc_c
    )
    prop_trees = np.clip(prop_trees, 0, 1)

    return prop_trees


class C3C4Competition:
    r"""Implementation of the C3/C4 competition model.

    This class provides an implementation of the calculations of C3/C4 competition,
    described by :cite:t:`lavergne:2020a`. The key inputs ``ggp_c3`` and ``gpp_c4`` are
    gross primary productivity (GPP) estimates for C3 or C4 pathways `alone`  using the
    :class:`~pyrealm.pmodel.pmodel.PModel`

    These estimates are used to calculate the relative advantage of C4 over C3
    photosynthesis (:math:`A_4`), the expected fraction of C4 plants in the community
    (:math:`F_4`) and hence fraction of GPP from C4 plants as follows:

    1. The proportion advantage in GPP for C4 plants is calculated as:

        .. math::
            :nowrap:

            \[
            A_4 = \frac{\text{GPP}_{C4} - \text{GPP}_{C3}}{\text{GPP}_{C3}}
            \]

    2. The proportion GPP advantage :math:`A_4` is converted to an expected fraction of
       C4 :math:`F_4` plants using the function
       :func:`~pyrealm.pmodel.competition.convert_gpp_advantage_to_c4_fraction`.

    3. A model of tree cover from C3 trees is then used to penalise the fraction of C4
       plants due to shading. The function
       :func:`~pyrealm.pmodel.competition.calculate_tree_proportion` is used to estimate
       the proportion (:math:`h`) and the C4 fraction is then discounted as :math:`F_4 =
       F_4 (1 - h)`.

    4. Two masks are applied. First, :math:`F_4 = 0` in locations where the mean  air
       temperature of the coldest month is too low for C4 plants. Second, :math:`F_4` is
       set as unknown for croplands, where the fraction is set by agricultural
       management, not competition.

    Args:
        gpp_c3: Total annual GPP (gC m-2 yr-1) from C3 plants alone.
        gpp_c4: Total annual GPP (gC m-2 yr-1) from C4 plants alone.
        treecover: Percentage tree cover (%).
        below_t_min: A boolean mask, temperatures too low for C4 plants.
        cropland: A boolean mask indicating cropland locations.
        const: An instance of :class:`~pyrealm.constants.competition_const.C3C4Const`
            providing parameterisation for the competition model.
    """

    # Design Notes: see paper Lavergne et al. (submitted).
    #
    # DO (24/05/2022): I have separated out the functions for different steps
    # into private methods, partly to keep the code cleaner, partly with a
    # slightly hopeful idea that future users could substitute these functions
    # via subclassing, but _mostly_ because being able to access these functions
    # independently makes it much easier to document the steps.

    # TODO - could accept PModel instances for gpp_c3 and gpp_c4 and auto-scale
    #        gpp and check that they are c3 and c4 models.
    #      - Would also allow the estimate isotopic discrimination to work
    #        automatically.
    #      - Axis argument to aggregate values along a time axis?
    #        nansum for gpp  and nanmean for  DeltaC13/4_alone.

    def __init__(
        self,
        gpp_c3: NDArray,
        gpp_c4: NDArray,
        treecover: NDArray,
        below_t_min: NDArray,
        cropland: NDArray,
        const: C3C4Const = C3C4Const(),
    ):
        # Check inputs are congruent
        self.shape: tuple = check_input_shapes(
            gpp_c3, gpp_c4, treecover, cropland, below_t_min
        )
        self.const: C3C4Const = const

        # Step 1: calculate the percentage advantage in GPP of C4 plants from
        # annual total GPP estimates for C3 and C4 plants. This uses use
        # np.full to handle division by zero without raising warnings
        gpp_adv_c4 = np.full(self.shape, np.nan)
        self.gpp_adv_c4: NDArray = np.divide(
            gpp_c4 - gpp_c3, gpp_c3, out=gpp_adv_c4, where=gpp_c3 > 0
        )
        """The proportional advantage in GPP of C4 over C3 plants"""

        # Step 2: calculate the initial C4 fraction based on advantage modulated
        # by treecover.
        frac_c4 = convert_gpp_advantage_to_c4_fraction(
            self.gpp_adv_c4, treecover=treecover, const=const
        )

        # Step 3: calculate the proportion of trees shading C4 plants, scaling
        # the predicted GPP to kilograms.
        prop_trees = calculate_tree_proportion(gppc3=gpp_c3 / 1000, const=const)
        frac_c4 = frac_c4 * (1 - prop_trees)

        # Step 4: remove areas below minimum temperature
        # mypy - this is a short term fix awaiting better resolution of mixed scalar and
        #        array inputs.
        frac_c4[below_t_min] = 0  # type: ignore

        # Step 5: remove cropland areas
        frac_c4[cropland] = np.nan  # type: ignore

        self.frac_c4: NDArray = frac_c4
        """The estimated fraction of C4 plants."""

        self.gpp_c3_contrib: NDArray = gpp_c3 * (1 - self.frac_c4)
        """The estimated contribution of C3 plants to GPP (gC m-2 yr-1)"""
        self.gpp_c4_contrib = gpp_c4 * self.frac_c4
        """The estimated contribution of C4 plants to GPP (gC m-2 yr-1)"""

        # Define attributes used elsewhere
        self.Delta13C_C3: NDArray
        r"""Contribution from C3 plants to (:math:`\Delta\ce{^13C}`, permil)."""
        self.Delta13C_C4: NDArray
        r"""Contribution from C4 plants to (:math:`\Delta\ce{^13C}`, permil)."""
        self.d13C_C3: NDArray
        r"""Contribution from C3 plants to (:math:`d\ce{^13C}`, permil)."""
        self.d13C_C4: NDArray
        r"""Contribution from C3 plants to (:math:`d\ce{^13C}`, permil)."""

    def __repr__(self) -> str:
        """Generates a string representation of a C3C4Competition instance."""
        return f"C3C4Competition(shape={self.shape})"

    def estimate_isotopic_discrimination(
        self, d13CO2: NDArray, Delta13C_C3_alone: NDArray, Delta13C_C4_alone: NDArray
    ) -> None:
        r"""Estimate CO2 isotopic discrimination values.

        Creating an instance of :class:`~pyrealm.pmodel.isotopes.CalcCarbonIsotopes`
        from a :class:`~pyrealm.pmodel.pmodel.PModel` instance provides estimated total
        annual descrimination against Carbon 13 (:math:`\Delta\ce{^13C}`) for a single
        photosynthetic pathway.

        This method allows predictions from C3 and C4 pathways to be combined to
        calculate the contribution from C3 and C4 plants given the estimated fraction of
        C4 plants. It also calculates the contributions to annual stable carbon isotopic
        composition (:math:`d\ce{^13C}`).

        Calling this method populates the attributes
        :attr:`~pyrealm.pmodel.competition.C3C4Competition.Delta13C_C3`,
        :attr:`~pyrealm.pmodel.competition.C3C4Competition.Delta13C_C4`,
        :attr:`~pyrealm.pmodel.competition.C3C4Competition.d13C_C3`, and
        :attr:`~pyrealm.pmodel.competition.C3C4Competition.d13C_C4`.

        Args:
            d13CO2: stable carbon isotopic composition of atmospheric CO2
                (permil)
            Delta13C_C3_alone: annual discrimination against 13C for C3
                plants (permil)
            Delta13C_C4_alone: annual discrimination against 13C for C4
                plants (permil)
        """

        _ = check_input_shapes(
            self.gpp_adv_c4, d13CO2, Delta13C_C3_alone, Delta13C_C4_alone
        )

        self.Delta13C_C3 = Delta13C_C3_alone * (1 - self.frac_c4)
        self.Delta13C_C4 = Delta13C_C4_alone * self.frac_c4

        self.d13C_C3 = (d13CO2 - self.Delta13C_C3) / (1 + self.Delta13C_C3 / 1000)
        self.d13C_C4 = (d13CO2 - self.Delta13C_C4) / (1 + self.Delta13C_C4 / 1000)

    def summarize(self, dp: int = 2) -> None:
        """Print summary of estimates of C3/C4 competition.

        Prints a summary of the calculated values in a C3C4Competition instance
        including the mean, range and number of nan values. This will always show
        fraction of C4 and GPP estaimates and isotopic estimates are shown if
        :meth:`~pyrealm.pmodel.competition.C3C4Competition.estimate_isotopic_discrimination`
        has been run.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [
            ("frac_c4", "-"),
            ("gpp_c3_contrib", "gC m-2 yr-1"),
            ("gpp_c4_contrib", "gC m-2 yr-1"),
        ]

        if hasattr(self, "d13C_C3"):
            attrs.extend(
                [
                    ("Delta13C_C3", "permil"),
                    ("Delta13C_C4", "permil"),
                    ("d13C_C3", "permil"),
                    ("d13C_C4", "permil"),
                ]
            )

        summarize_attrs(self, attrs, dp=dp)
