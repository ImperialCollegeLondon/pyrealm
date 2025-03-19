"""Implements a two-leaf, two-stream canopy model for Pyrealm.

This module provides functionality to partition irradiance into sunlit and shaded
fractions of the canopy (via ``TwoLeafIrradience``) and to estimate gross primary
productivity (via ``TwoLeafAssimilation``). The approach follows De Pury and Farquhar
(1997) and aligns closely with the two-leaf methodology used in the BESS model.
"""

import numpy as np
from numpy.typing import NDArray

from pyrealm.core.utilities import check_input_shapes
from pyrealm.constants.core_const import CoreConst
from pyrealm.constants.two_leaf_canopy import TwoLeafConst
from pyrealm.pmodel.optimal_chi import OptimalChiABC
from pyrealm.pmodel.pmodel import PModel, SubdailyPModel


class TwoLeafIrradience:
    """Running the two leaf, two stream model within Pyrealm.

    This class implements the :cite:t:`depury:1997a` two leaf, two stream model. This
    model is chosen to
    provide a better representation than the big leaf model and to align closely to
    the workings of the ``BESS`` model :cite:alp:`ryu:2011a`.

    The public :meth:`calc_absorbed_irradiance` is used to generate values for the
    variables associated with irradiance.

    The calculated values are used in the estimation of gross primary productivity
    (``GPP``). An instance of this class is accepted by the
    :class:`TwoLeafAssimilation`, which uses these results to calulate ``GPP``.

    Args:
        solar_elevation: Array of solar elevation angles (radians).
        ppfd: Array of photosynthetic photon flux density values (µmol m-2 s-1).
        leaf_area_index: Array of leaf area index values.
        patm: Array of atmospheric pressure values (pascals).
        core_constants: An instance of the core constants class.
        two_leaf_constants: An instance of the two leaf constants class.
    """

    def __init__(
        self,
        solar_elevation: NDArray[np.float64],
        ppfd: NDArray[np.float64],
        leaf_area_index: NDArray[np.float64],
        patm: NDArray[np.float64],
        core_constants: CoreConst = CoreConst(),
        two_leaf_constants: TwoLeafConst = TwoLeafConst(),
    ):
        # Check shapes are consistent
        check_input_shapes(solar_elevation, ppfd, leaf_area_index, patm)

        self.solar_elevation: NDArray[np.float64] = solar_elevation
        r"""The solar elevation inputs (:math:`\beta`, radians)"""
        self.ppfd: NDArray[np.float64] = ppfd
        """The photosynthetic photon flux density inputs (PPFD, µmol m-2 s-1)"""
        self.leaf_area_index: NDArray[np.float64] = leaf_area_index
        """The leaf area index inputs (:math:`L`, unitless)"""
        self.patm: NDArray[np.float64] = patm
        """The atmospheric pressure (:math:`P`, pascals)"""
        self.core_constants: CoreConst = core_constants
        """An instance of the core constants class."""
        self.two_leaf_constants: TwoLeafConst = two_leaf_constants
        """An instance of the two leaf constants class."""

        self.beam_extinction_coefficient: NDArray[np.float64]
        """The beam extinction coefficient (:math:`k_{b}`)"""
        self.scattered_beam_extinction_coefficient: NDArray[np.float64]
        """The scattered beam extinction coefficient (:math:`k_{b}`)"""
        self.fraction_of_diffuse_radiation: NDArray[np.float64]
        """The fraction of diffuse radiation (:math:`f_d`)"""
        self.horizontal_leaf_beam_irradiance: float
        r"""The beam irradiance for horizontal leaves (:math:`\rho_h`)"""
        self.uniform_leaf_beam_irradiance: NDArray[np.float64]
        r"""The beam irradiance for leaves with uniform angle distribution
         (:math:`\rho_{cb}`)"""
        self.diffuse_irradiance: NDArray[np.float64]
        """The diffuse irradiance (:math:`I_d`) reaching the canopy."""
        self.beam_irradiance: NDArray[np.float64]
        """The beam irradiance (:math:`I_b`) reaching the canopy."""
        self.canopy_irradiance: NDArray[np.float64]
        """The total canopy irradiance (:math:`I_c`)."""
        self.sunlit_beam_irradiance: NDArray[np.float64]
        """The sunlit beam irradiance (:math:`I_{sb}`)."""
        self.sunlit_diffuse_irradiance: NDArray[np.float64]
        """The sunlit diffuse irradiance (:math:`I_{sd}`)."""
        self.sunlit_scattered_irradiance: NDArray[np.float64]
        """The sunlit scattered irradiance (:math:`I_{ss}`)."""
        self.sunlit_absorbed_irradiance: NDArray[np.float64]
        """The sunlit leaf absorbed irradiance (:math:`I_{csun})."""
        self.shaded_absorbed_irradiance: NDArray[np.float64]
        """The shaded leaf absorbed irradiance (:math:`I_{cshade})."""

    #     self.no_NaNs: bool = self._check_for_NaN()
    #     self.no_negatives: bool = self._check_for_negative_values()

    # def _check_for_NaN(self) -> bool:
    #     """Tests for any NaN in input arrays."""
    #     arrays = [self.beta_angle, self.ppfd, self.leaf_area_index, self.patm]
    #     try:
    #         for array in arrays:
    #             if np.isnan(array).any():
    #                 raise ValueError("Nan data in input array")
    #         return True
    #     except Exception as e:
    #         print(f"Error in input NaN check: {e}")
    #         return False

    # def _check_for_negative_values(self) -> bool:
    #     """Tests for negative values in arrays."""
    #     try:
    #         for array in [self.ppfd, self.leaf_area_index, self.patm]:
    #             if not (array >= 0).all():
    #                 raise ValueError("Input arrays contain negative values.")
    #         return True
    #     except Exception as e:
    #         print(f"Error in input negative number check: {e}")
    #         return False

    def calc_absorbed_irradiance(self) -> None:
        r"""Calculate absorbed irradiance for sunlit and shaded leaves.

        The internal function calls are as follows:

        1. Calculate the beam extinction coefficient :math:`k_{b}` using
        :func:`beam_extinction_coeff`

        2. Calculate the scattered beam extinction coefficient using
        :func:`scattered_beam_extinction_coeff`

        3. Calculate fraction of diffuse radiation using :func:`fraction_of_diffuse_rad`

        4. Calculate beam irradience for horizontal leaves using
        :func:`beam_irradience_h_leaves`

        5. Calculate the beam irradience for a uniform leaf angle distribution with
        :func:`beam_irrad_unif_leaf_angle_dist`

        6. Calculate diffuse radiation with :func:`diffuse_radiation`

        7. Calculate direct beam irradience with :func:`beam_irradience`

        8. Calculate scattered beam irradience with :func:`scattered_beam_irradience`

        9. Calculate the total canopy irradience with :func:`canopy_irradience`

        10. Calculate the sunlit beam irradience with :func:`sunlit_beam_irrad`

        11. Calulate the sunlit diffuse irradience with :func:`sunlit_diffuse_irrad`

        12. Calculate the scattered irradience recieved by sunlit portion of canopy with
        :func:`sunlit_scattered_irrad`

        13. Calulate total irradience absorbed by sunlit portion of canopy with
        :func:`sunlit_absorbed_irrad`

        14. Calculate the irradience absorbed by the shaded farction of the canopy with
        :func:`shaded_absorbed_irrad`

        """

        # Calculate the beam extinction coefficient for direct light
        self.beam_extinction_coefficient = calculate_beam_extinction_coefficient(
            solar_elevation=self.solar_elevation,
            solar_obscurity_angle=self.two_leaf_constants.solar_obscurity_angle,
            extinction_numerator=self.two_leaf_constants.direct_beam_extinction_numerator,
        )

        # Calculate the extinction coefficient for scattered light
        self.scattered_beam_extinction_coefficient = calculate_beam_extinction_coefficient(
            solar_elevation=self.solar_elevation,
            solar_obscurity_angle=self.two_leaf_constants.solar_obscurity_angle,
            extinction_numerator=self.two_leaf_constants.scattered_beam_extinction_numerator,
        )

        # Calculate the fraction of diffuse radiation
        self.fraction_of_diffuse_radiation = calculate_fraction_of_diffuse_radiation(
            patm=self.patm,
            solar_elevation=self.solar_elevation,
            standard_pressure=self.core_constants.k_Po,
            atmospheric_scattering=self.two_leaf_constants.atmospheric_scattering_coefficient,
            leaf_diffusion_factor=self.two_leaf_constants.leaf_diffusion_factor,
        )

        # Calculate the horizontal leaf beam irradiance and hence the beam irradiance
        # for leaves following a uniform angle distribution.
        self.horizontal_leaf_beam_irradiance = calculate_beam_irradiance_horizontal_leaves(
            scattering_coefficient=self.two_leaf_constants.leaf_scattering_coefficient
        )

        self.uniform_leaf_beam_irradiance = calculate_beam_irradiance_uniform_leaves(
            beam_extinction=self.beam_extinction_coefficient,
            beam_irradiance_horizontal_leaves=self.horizontal_leaf_beam_irradiance,
        )

        # Calculate the diffuse and beam irradiance reaching the canopy
        self.diffuse_irradiance = calculate_diffuse_irradiance(
            diffuse_fraction=self.fraction_of_diffuse_radiation,
            ppfd=self.ppfd,
        )
        self.beam_irradiance = calculate_beam_irradiance(
            diffuse_fraction=self.fraction_of_diffuse_radiation,
            ppfd=self.ppfd,
        )

        # Calculate canopy irradiance
        self.canopy_irradiance = calculate_canopy_irradiance(
            uniform_leaf_beam_irradiance=self.uniform_leaf_beam_irradiance,
            beam_irradiance=self.beam_irradiance,
            scattered_beam_extinction_coefficient=self.scattered_beam_extinction_coefficient,
            diffuse_radiation=self.diffuse_irradiance,
            leaf_area_index=self.leaf_area_index,
            canopy_reflection_coefficient=self.two_leaf_constants.canopy_reflection_coefficient,
        )

        # Calculate fractions of the sunlit leaf irradiance
        self.sunlit_beam_irradiance = calculate_sunlit_beam_irradiance(
            self.beam_irradiance,
            self.two_leaf_constants.leaf_scattering_coefficient,
            self.beam_extinction_coefficient,
            self.leaf_area_index,
        )

        self.sunlit_diffuse_irradiance = calculate_sunlit_diffuse_irradiance(
            self.diffuse_irradiance,
            self.two_leaf_constants.canopy_reflection_coefficient,
            self.two_leaf_constants.diffuse_extinction_coefficient,
            self.beam_extinction_coefficient,
            self.leaf_area_index,
        )

        self.sunlit_scattered_irradiance = calculate_sunlit_scattered_irradiance(
            self.beam_irradiance,
            self.uniform_leaf_beam_irradiance,
            self.scattered_beam_extinction_coefficient,
            self.beam_extinction_coefficient,
            self.leaf_area_index,
            self.two_leaf_constants.canopy_reflection_coefficient,
        )

        self.sunlit_absorbed_irradiance = calculate_sunlit_absorbed_irradiance(
            self.sunlit_beam_irradiance,
            self.sunlit_diffuse_irradiance,
            self.sunlit_scattered_irradiance,
        )
        self.shaded_absorbed_irradiance = calculate_shaded_absorbed_irradiance(
            self.solar_elevation,
            self.two_leaf_constants.solar_obscurity_angle,
            self.canopy_irradiance,
            self.sunlit_absorbed_irradiance,
        )


class TwoLeafAssimilation:
    """A class to estimate gross primary production (``GPP``) using a two-leaf approach.

    This class integrates a photosynthesis model
    (:class:`~pyrealm.pmodel.pmodel.PModel` or
    :class:`~pyrealm.pmodel.pmodel.SubdailyPModel`) and irradiance data
    (:class:`~pyrealm.pmodel.two_leaf.TwoLeafIrradience`) to compute
    various canopy photosynthetic properties and ``GPP`` estimates.

    Args:
        pmodel (PModel | SubdailyPModel): The photosynthesis model used for
            assimilation.
        irrad (TwoLeafIrradience): Irradiance data required for ``GPP`` calculations.
        leaf_area_index (NDArray): Array of leaf area index values.
    """

    def __init__(
        self,
        pmodel: PModel | SubdailyPModel,
        irrad: TwoLeafIrradience,
    ):
        """Initialize the ``TwoLeafAssimilation`` class.

        Args:
            pmodel (PModel | SubdailyPModel): The photosynthesis model.
            irrad (TwoLeafIrradience): The irradiance data.
            leaf_area_index (NDArray): Array of leaf area index values.
        """
        self.pmodel = pmodel
        self.irrad = irrad

        self.vcmax_pmod: NDArray = self.pmodel.vcmax
        self.vcmax25_pmod: NDArray = self.pmodel.vcmax25
        self.optchi_obj: OptimalChiABC = self.pmodel.optchi
        self.core_const: CoreConst = self.pmodel.core_const

        self.gpp: NDArray

    def gpp_estimator(self) -> None:
        """Estimate the gross primary production (``GPP``) using the two-leaf model.

        This method uses the following functions to calculate the ``GPP`` estimate,
        including carboxylation rates, assimilation rates, and electron transport
        rates for sunlit and shaded leaves. Ultimately leading to the estimation of
        ``GPP``.

        **Calculation Steps:**

        1. Calculate the canopy extinction coefficient with
        :func:`beam_extinction_coeff`

        2. Calculate the canopy carboxylation rate with
        :func:`Vmax25_canopy`

        3. Calculate the carboxylation rate in sunlit areas
        :func:`Vmax25_sun`

        4. Calculate the carboxylation rate in shaded areas
        :func:`Vmax25_shade`

        5. Calculate the sunlit and shaded carboxylation rates scaled by temperature
        with :func:`carboxylation_scaling_to_T`

        6. Calculate the sunlit and shaded photosynthetic rate
        with :func:`photosynthetic_estimate`

        7. Calculate the sunlit and shaded maximum rate of electron transport
        with :func:`Jmax25`

        8. Calculate the sunlit and shaded temperature corrected rate of electron
        transport with :func:`Jmax25_temp_correction`

        9. Calculate the sunlit and shaded electron transport rate
        :func:`electron_transport_rate`

        10. Calculate the sunlit and shaded assimilation rate driven by electron
        transport with :func:`assimilation_rate`

        11. Calculate the sunlit and shaded canopy assimilation
        with :func:`assimilation_canopy`

        12. Calculate the gross primary productivity
        with :func:`gross_primary_product`

        **Sets:**
        gpp_estimate (NDArray): Estimated gross primary production.
        """
        self.kv_Lloyd = canopy_extinction_coefficient(self.vcmax_pmod)

        self.Vmax25_canopy = Vmax25_canopy(
            self.irrad.leaf_area_index, self.vcmax25_pmod, self.kv_Lloyd
        )
        self.Vmax25_sun = Vmax25_sun(
            self.irrad.leaf_area_index, self.vcmax25_pmod, self.kv_Lloyd, self.irrad.kb
        )
        self.Vmax25_shade = Vmax25_shade(self.Vmax25_canopy, self.Vmax25_sun)

        self.Vmax_sun = carboxylation_scaling_to_T(self.Vmax25_sun, self.pmodel.env.tc)
        self.Vmax_shade = carboxylation_scaling_to_T(
            self.Vmax25_shade, self.pmodel.env.tc
        )

        self.Av_sun = photosynthetic_estimate(self.Vmax_sun, self.optchi_obj.mc)
        self.Av_shade = photosynthetic_estimate(self.Vmax_shade, self.optchi_obj.mc)

        self.Jmax25_sun = Jmax25(self.Vmax25_sun)
        self.Jmax25_shade = Jmax25(self.Vmax25_shade)

        self.Jmax_sun = Jmax25_temp_correction(self.Jmax25_sun, self.pmodel.env.tc)
        self.Jmax_shade = Jmax25_temp_correction(self.Jmax25_shade, self.pmodel.env.tc)

        self.J_sun = electron_transport_rate(self.Jmax_sun, self.irrad.I_csun)
        self.J_shade = electron_transport_rate(self.Jmax_shade, self.irrad.I_cshade)

        self.Aj_sun = assimilation_rate(self.optchi_obj.mj, self.J_sun)
        self.Aj_shade = assimilation_rate(self.optchi_obj.mj, self.J_shade)

        self.Acanopy_sun = assimilation_canopy(
            self.Aj_sun, self.Av_sun, self.irrad.beta_angle, self.irrad.k_sol_obs_angle
        )
        self.Acanopy_shade = assimilation_canopy(
            self.Aj_shade,
            self.Av_shade,
            self.irrad.beta_angle,
            self.irrad.k_sol_obs_angle,
        )

        self.gpp_estimate = gross_primary_product(
            self.core_const.k_c_molmass, self.Acanopy_sun, self.Acanopy_shade
        )


def calculate_beam_extinction_coefficient(
    solar_elevation: NDArray,
    solar_obscurity_angle: float = TwoLeafConst().solar_obscurity_angle,
    extinction_numerator: float = TwoLeafConst().direct_beam_extinction_numerator,
) -> NDArray:
    r"""Calculate the beam extinction coefficient.

    The beam extinction coefficient (:math:`k_b`) captures changes in the the
    attenuation of direct sunlight through the canopy with variation in the solar
    elevation angle.

    .. math::

        k_b = \frac{n}{\sin \max{\beta, \beta_{ob}}}

    Args:
        solar_elevation: Array of solar elevation angles (:math:`\beta`)
        solar_obscurity_angle: Threshold angle at which the beam extinction coefficient
            reaches a maximum (:math:`\beta_{ob}`).
        extinction_numerator: The numerator used in the calculation (:math:`n`).

    Returns:
        An array of beam extinction coefficients.
    """

    # TODO - checking this with Keith
    # max_val = extinction_numerator / np.sin(solar_obscurity_angle)
    if extinction_numerator == 0.5:
        max_val = np.array([30])
    elif extinction_numerator == 0.46:
        max_val = np.array([27])

    return np.where(
        solar_elevation > solar_obscurity_angle,
        extinction_numerator / np.sin(solar_elevation),
        max_val,  # extinction_numerator / np.sin(solar_obscurity_angle)
    )


def calculate_fraction_of_diffuse_radiation(
    patm: NDArray,
    solar_elevation: NDArray,
    standard_pressure: float = CoreConst().k_Po,
    atmospheric_scattering: float = TwoLeafConst().atmospheric_scattering_coefficient,
    leaf_diffusion_factor: float = TwoLeafConst().leaf_diffusion_factor,
) -> NDArray:
    r"""Calculate the fraction of diffuse radiation.

    The fraction of diffuse radiation (:math:`f_d`) captures the proportion of
    sunlight that is scattered before reaching the canopy, following equation A23 and
    A25 of :cite:t:`depury:1997a`.

    .. math::
        :nowrap:

        \[
            \begin{align*}
                m &= \left(P / P_0\right) / \sin \beta \\
                f_d &= \frac{1 - a^m}{1 + a^m \, \left(1 / f_a - 1\right)
            \end{align*}
        \]


    Args:
        patm: Atmospheric pressure values (:math:`P`, pascals).
        solar_elevation: Solar elevation angles (:math:\beta`, radians).
        standard_pressure: Standard atmospheric pressure (:math:`P_0`, pascals).
        atmospheric_scattering: Atmospheric scattering factor (:math:`f_a`).
        leaf_diffusion_factor: Leaf derived diffusion factor (:math:`a`).

    Returns:
        Array of fractions of diffuse radiation.
    """

    # Optical air mass
    m = (patm / standard_pressure) / np.sin(solar_elevation)

    return (1 - leaf_diffusion_factor**m) / (
        1 + (leaf_diffusion_factor**m * (1 / atmospheric_scattering - 1))
    )


def calculate_beam_irradiance_horizontal_leaves(
    scattering_coefficient: float = TwoLeafConst.leaf_scattering_coefficient,
) -> float:
    r"""Calculate the beam irradiance for horizontal leaves.

    The beam irradiance for horizontal leaves (:math:`\rho_h`) considers the leaf
    orientation and the direct sunlight received, following equation A20 of
    :cite:t:`depury:1997a`.

    .. math::

        \rho_h = \frac{1 - \sqrt{1 - \sigma}}{1 + \sqrt{1 - \sigma}}

    Args:
        scattering_coefficient: Scattering coefficient associated with horizontal
            leaves (:math:`\sigma`).

    Returns:
        A float constant giving the beam irradiances for horizontal leaves.
    """

    return (1 - np.sqrt(1 - scattering_coefficient)) / (
        1 + np.sqrt(1 - scattering_coefficient)
    )


def calculate_beam_irradiance_uniform_leaves(
    beam_extinction: NDArray[np.float64],
    beam_irradiance_horizontal_leaves: float,
) -> NDArray:
    r"""Calculate the beam irradiance for leaves with a uniform angle distribution.

    The beam irradiance with a uniform leaf angle distribution (:math:`\rho_{cb}`)
    captures different leaf orientations within the canopy, following equation A19 of
    :cite:t:`depury:1997a`

    .. math::

        \rho_{cb} = 1 - \exp \left(-\frac{2 \rho_h \, k_b}{1 + k_b}\right)

    Args:
        beam_extinction: Array of beam extinction coefficients (:math:`k_b`).
        beam_irradiance_horizontal_leaves: The beam irradiance for horizontal leaves
            (:math:`\rho_h`).

    Returns:
        Array of beam irradiances.
    """

    return 1 - np.exp(
        -2 * beam_irradiance_horizontal_leaves * beam_extinction / (1 + beam_extinction)
    )


def calculate_diffuse_irradiance(diffuse_fraction: NDArray, ppfd: NDArray) -> NDArray:
    r"""Calculate the diffuse radiation.

    The diffuse irradiance (:math:`I_d`) is the portion of sunlight that is scattered in
    the atmosphere before reaching the canopy.

    .. math::

        I_d = \max{0, \textrm{PPFD} \cdot f_d}

    Args:
        diffuse_fraction: The fraction of diffuse irradiance (:math:`f_d`).
        ppfd: The photosynthetic photon flux density.

    Returns:
        Array of diffuse light irradiance values.
    """

    return np.clip(ppfd * diffuse_fraction, a_min=0, a_max=np.inf)


def calculate_beam_irradiance(diffuse_fraction: NDArray, ppfd: NDArray) -> NDArray:
    r"""Calculate the beam irradiance.

    The beam irradiance (`:math:`I_b`)is the direct component of sunlight that reaches
    the canopy without being scattered.

    .. math::

        I_b = \text{ppfd} \cdot (1 - \text{fd})

    Args:
        diffuse_fraction: The fraction of diffuse radiation (:math:`f_d`).
        ppfd: The photosynthetic photon flux density.

    Returns:
        Array of direct beam irradiance values.
    """

    return ppfd * (1 - diffuse_fraction)


def calculate_scattered_beam_irradiance(
    I_b: NDArray,
    kb: NDArray,
    kb_prime: NDArray,
    rho_cb: NDArray,
    leaf_area_index: NDArray,
    k_sigma: NDArray,
) -> NDArray:
    r"""Calculate the scattered beam irradiance.

    The scattered beam irradiance (:math:`I_bs`) is the portion of direct sunlight that
    is scattered within the canopy.

    .. math::

        I_{bs} = I_b \cdot (1 - \rho_{cb}) \cdot kb_{prime} \cdot \exp(-kb_{prime} \cdot
        \text{leaf_area_index}) - (1 - k_{sigma}) \cdot kb \cdot \exp(-kb \cdot
        \text{leaf_area_index})

    Args:
        I_b (NDArray): Array of beam irradiance values.
        kb (NDArray): Array of beam extinction coefficients.
        kb_prime (NDArray): Array of scattered beam extinction coefficients.
        rho_cb (NDArray): Array of beam irradiances with uniform leaf angle
            distribution.
        leaf_area_index (NDArray): Array of leaf area index values.
        k_sigma (NDArray): Array of sigma values.

    Returns:
        NDArray: Array of scattered beam irradiance values.
    """

    I_bs = I_b * (1 - rho_cb) * kb_prime * np.exp(-kb_prime * leaf_area_index) - (
        1 - k_sigma
    ) * kb * np.exp(-kb * leaf_area_index)

    return I_bs


def calculate_canopy_irradiance(
    uniform_leaf_beam_irradiance: NDArray,
    beam_irradiance: NDArray,
    scattered_beam_extinction_coefficient: NDArray,
    diffuse_radiation: NDArray,
    leaf_area_index: NDArray,
    canopy_reflection_coefficient: float = TwoLeafConst().canopy_reflection_coefficient,
) -> NDArray:
    r"""Calculate the canopy irradiance.

    The canopy irradiance (:math:`I_c`) is the total irradiance within the canopy,
    including both direct and diffuse radiation components.

    .. math::

        I_c = (1 - \rho_{cb})  I_b   (1 - \exp(-k_b' L)) +
            (1 - \rho{cd})  I_d  (1 - \exp(-k_b' L))

    Args:
        uniform_leaf_beam_irradiance : Array of beam irradiances with uniform leaf angle
            distribution (:math:`\rho_{cb}`).
        beam_irradiance : Array of beam irradiance values (:math:`I_b`).
        scattered_beam_extinction_coefficient : Array of scattered beam extinction
            coefficients (:math:`k_b'`).
        diffuse_radiation : Array of diffuse radiation values (:math:`I_d`).
        leaf_area_index : Array of leaf area index values (:math:`L`).
        canopy_reflection_coefficient : Canopy_reflection_coefficient
            (:math:`\rho_{cd}`).

    Returns:
        NDArray: Array of canopy irradiance values.
    """

    return (1 - uniform_leaf_beam_irradiance) * beam_irradiance * (
        1 - np.exp(-scattered_beam_extinction_coefficient * leaf_area_index)
    ) + (1 - canopy_reflection_coefficient) * diffuse_radiation * (
        1 - np.exp(-scattered_beam_extinction_coefficient * leaf_area_index)
    )


def calculate_sunlit_beam_irradiance(
    I_b: NDArray, k_sigma: float, kb: NDArray, leaf_area_index: NDArray
) -> NDArray:
    r"""Calculate the sunlit beam irradiance.

    The sunlit beam irradiance (:math:`Isun_beam`) is the direct sunlight received by
    the sunlit portion of the canopy.

    .. math::

        I_{sun_beam} = I_b \cdot (1 - k_{sigma}) \cdot (1 - \exp(-kb \cdot
        \text{leaf_area_index}))

    Args:
        I_b (NDArray): Array of beam irradiance values.
        k_sigma (float): Constant for calculating the sunlit beam irradiance.
        kb (NDArray): Array of beam extinction coefficients.
        leaf_area_index (NDArray): Array of leaf area index values.

    Returns:
        NDArray: Array of sunlit beam irradiance values.
    """
    return I_b * (1 - k_sigma) * (1 - np.exp(-kb * leaf_area_index))


def calculate_sunlit_diffuse_irradiance(
    I_d: NDArray,
    k_rho_cd: float,
    k_kd_prime: float,
    kb: NDArray,
    leaf_area_index: NDArray,
) -> NDArray:
    r"""Calculate the sunlit diffuse irradiance.

    The sunlit diffuse irradiance (:math:`Isun_diffuse`) is the diffuse radiation
    received by the sunlit portion of the canopy.

    .. math::

        I_{sun_diffuse} = I_d \cdot (1 - \text{k_{rho_cd}}) \cdot
        (1 - \exp(-(k_{kd_prime} + kb) \cdot
        \text{leaf_area_index})) \cdot \frac{k_{kd_prime}}{k_{kd_prime} + kb}

    Args:
        I_d (NDArray): Array of diffuse radiation values.
        k_rho_cd (float): Constant rho_cd value.
        k_kd_prime (float): Constant for calculating the sunlit diffuse irradiance.
        kb (NDArray): Array of beam extinction coefficients.
        leaf_area_index (NDArray): Array of leaf area index values.

    Returns:
        NDArray: Array of sunlit diffuse irradiance values.
    """
    return (
        I_d
        * (1 - k_rho_cd)
        * (1 - np.exp(-(k_kd_prime + kb) * leaf_area_index))
        * k_kd_prime
        / (k_kd_prime + kb)
    )


def calculate_sunlit_scattered_irradiance(
    I_b: NDArray,
    rho_cb: NDArray,
    kb_prime: NDArray,
    kb: NDArray,
    leaf_area_index: NDArray,
    k_sigma: float,
) -> NDArray:
    r"""Calculate the sunlit scattered irradiance.

    The sunlit scattered irradiance (:math:`Isun_scattered`) is the scattered radiation
    received by the sunlit portion of the canopy.

    .. math::

        I_{sun_scattered} = I_b \cdot ((1 - \rho_{cb}) \cdot
        (1 - \exp(-(kb_{prime} + kb) \cdot
        \text{leaf_area_index})) \cdot \frac{kb_{prime}}{kb_{prime} + kb}
        - (1 - k_{sigma})
        \cdot (1 - \exp(-2 \cdot kb \cdot \text{leaf_area_index})) / 2)

    Args:
        I_b (NDArray): Array of beam irradiance values.
        rho_cb (NDArray): Array of beam irradiances with uniform leaf angle
            distribution.
        kb_prime (NDArray): Array of scattered beam extinction coefficients.
        kb (NDArray): Array of beam extinction coefficients.
        leaf_area_index (NDArray): Array of leaf area index values.
        k_sigma (float): Constant for calculating the sunlit scattered irradiance.

    Returns:
        NDArray: Array of sunlit scattered irradiance values.
    """

    return I_b * (
        (1 - rho_cb)
        * (1 - np.exp(-(kb_prime + kb) * leaf_area_index))
        * kb_prime
        / (kb_prime + kb)
        - (1 - k_sigma) * (1 - np.exp(-2 * kb * leaf_area_index)) / 2
    )


def calculate_sunlit_absorbed_irradiance(
    Isun_beam: NDArray, Isun_diffuse: NDArray, Isun_scattered: NDArray
) -> NDArray:
    r"""Calculate the sunlit absorbed irradiance.

    The sunlit absorbed irradiance (:math:`I_csun`) is the total irradiance absorbed by
    the sunlit portion of the canopy, combining beam, diffuse, and scattered irradiance.

    .. math::

        I_{csun} = I_{sun-beam} + I_{sun-diffuse} + I_{sun-scattered}

    Args:
        Isun_beam (NDArray): Array of sunlit beam irradiance values.
        Isun_diffuse (NDArray): Array of sunlit diffuse irradiance values.
        Isun_scattered (NDArray): Array of sunlit scattered irradiance values.

    Returns:
        NDArray: Array of sunlit absorbed irradiance values.
    """
    return Isun_beam + Isun_scattered + Isun_diffuse


def calculate_shaded_absorbed_irradiance(
    beta_angle: NDArray, k_sol_obs_angle: float, I_c: NDArray, I_csun: NDArray
) -> NDArray:
    r"""Calculate the irradiance absorbed by the shaded fraction of the canopy.

    The irradiance absorbed by the shaded fraction of the canopy (:math:`I_cshade`) is
    calculated by subtracting the sunlit absorbed irradiance from the total canopy
    irradiance.

    .. math::

        I_{cshade} = \operatorname{max}(0, I_c - I_{csun})

    Args:
        beta_angle (NDArray): Array of solar elevation angles.
        k_sol_obs_angle (float): Solar angle threshold.
        I_c (NDArray): Array of canopy irradiance values.
        I_csun (NDArray): Array of sunlit absorbed irradiance values.

    Returns:
        NDArray: Array of irradiance absorbed by the shaded fraction of the canopy.
    """
    return np.where(beta_angle > k_sol_obs_angle, I_c - I_csun, 0)


def calculate_canopy_extinction_coefficient(vcmax_pmod: NDArray) -> NDArray:
    r"""Calculate :math:`k_v` parameter.

    This function calculates the extinction coefficient, :math:`k_v`, which represents
    how the photosynthetic capacity (Vcmax) decreases with depth in the plant canopy.
    The exponential model used here is derived from empirical data and represents how
    light attenuation affects photosynthetic capacity vertically within the canopy.

    Equation is sourced from Figure 10 of Lloyd et al. (2010).

    .. math::
        kv_{Lloyd} = \exp(0.00963 \cdot vcmax_{pmod} - 2.43)

    Note:
        ``Vcmax`` is used here rather than ``Vcmax_25``.

    Args:
        vcmax_pmod (NDArray): The ``Vcmax`` parameter for the pmodel.

    Returns:
        NDArray: The calculated :math:`kv_Lloyd` values.
    """
    kv_Lloyd = np.exp(0.00963 * vcmax_pmod - 2.43)
    return kv_Lloyd


def Vmax25_canopy(
    leaf_area_index: NDArray, vcmax25_pmod: NDArray, kv: NDArray
) -> NDArray:
    r"""Calculate carboxylation rate, :math:`V_max25` in the canopy at 25C.

    This function calculates the maximum carboxylation rate of the canopy at a
    reference temperature of 25°C. It integrates the vertical gradient of
    photosynthetic capacity across the leaf area index (``LAI``), considering the
    extinction coefficient for light.

    .. math::
        Vmax25_{canopy} = \text{LAI} \cdot vcmax25_{pmod} \cdot
        \left(\frac{1 - \exp(-\text{kv})}{\text{kv}}\right)


    Args:
        leaf_area_index (NDArray): The ``leaf area index``.
        vcmax25_pmod (NDArray): The ``Vcmax25`` parameter for the pmodel.
        kv (NDArray): The kv parameter.

    Returns:
        NDArray: The calculated Vmax25 canopy values.
    """
    Vmax25_canopy = leaf_area_index * vcmax25_pmod * ((1 - np.exp(-kv)) / kv)

    return Vmax25_canopy


def Vmax25_sun(
    leaf_area_index: NDArray, vcmax25_pmod: NDArray, kv: NDArray, kb: NDArray
) -> NDArray:
    r"""Calculate carboxylation in sunlit areas, :math:`Vmax25_sun` at 25C.

    This function calculates the maximum carboxylation rate for the sunlit portions of
    the canopy at 25°C. It considers both the extinction coefficient and the direct
    sunlight penetration parameter :math:`k_b`.

    .. math::

        Vmax25_{sun} = \text{leaf_area_index} \cdot vcmax25_{pmod} \cdot
        \left( \frac{1 - \exp(-\text{kv} - \text{kb} \cdot \text{leaf_area_index})}
        {\text{kv} + \text{kb} \cdot \text{leaf_area_index}} \right)

    Args:
        leaf_area_index (NDArray): The leaf area index (LAI).
        vcmax25_pmod (NDArray): The Vcmax25 parameter for the pmodel.
        kv (NDArray): The kv parameter.
        kb (NDArray): The irradiation kb parameter.

    Returns:
        NDArray: The calculated Vmax25 sun values.
    """
    Vmax25_sun = (
        leaf_area_index
        * vcmax25_pmod
        * ((1 - np.exp(-kv - kb * leaf_area_index)) / (kv + kb * leaf_area_index))
    )

    return Vmax25_sun


def Vmax25_shade(Vmax25_canopy: NDArray, Vmax_25_sun: NDArray) -> NDArray:
    r"""Calculate carboxylation in shaded areas, :math:`Vmax25_shade` at 25C.

    This function calculates the maximum carboxylation rate for the shaded portions of
    the canopy by subtracting the sunlit carboxylation rate from the total canopy
    carboxylation rate.

    .. math::

        Vmax25_{shade} = Vmax25_{canopy} - Vmax25_{sun}

    Args:
        Vmax25_canopy (NDArray): The ``Vmax25`` parameter for the canopy.
        Vmax_25_sun (NDArray): The ``Vmax25`` parameter for the sunlit areas.

    Returns:
        NDArray: The calculated Vmax25 shade values.
    """
    Vmax25_shade = Vmax25_canopy - Vmax_25_sun

    return Vmax25_shade


def carboxylation_scaling_to_T(Vmax25: NDArray, tc: NDArray) -> NDArray:
    r"""Convert carboxylation rates at 25C to rate at ambient temperature (C).

    This function adjusts the carboxylation rate from the reference temperature of 25°C
    to the ambient temperature using an Arrhenius-type function, which describes the
    temperature dependence of enzymatic reactions.

    .. math::
        Vmax_{sun} = \text{Vmax25} \cdot \exp \left(\frac{64800 \cdot (\text{tc}
        - 25)}{298 \cdot 8.314 \cdot (\text{tc} + 273)}\right)

    Args:
        Vmax25 (NDArray): The ``Vmax25`` parameter.
        tc (NDArray): The temperature in Celsius.

    Returns:
        NDArray: The carboxylation rates adjusted to ambient temperature.
    """

    Vmax = Vmax25 * np.exp(64800 * (tc - 25) / (298 * 8.314 * (tc + 273)))

    return Vmax


def photosynthetic_estimate(Vmax: NDArray, mc: NDArray) -> NDArray:
    r"""Calculate photosynthetic rate estimate, :math:`Av`.

    This function estimates photosynthetic rates by multiplying the carboxylation
    capacity by mc, the limitation term for Rubisco-limited assimilation.

    .. math::

        A_v = V_{max} \cdot mc

    Args:
        Vmax (NDArray): The ``Vmax`` parameter.
        mc (NDArray): limitation term for Rubisco-limited assimilation

    Returns:
        NDArray: The calculated photosynthetic estimates.
    """

    Av = Vmax * mc

    return Av


def Jmax25(Vmax25: NDArray) -> NDArray:
    r"""Calculate the maximum rate of electron transport :math:`Jmax25`.

    This function calculates the maximum rate of electron transport :math:`Jmax25`
    at 25°C, which is related to the capacity for light-driven electron transport in
    photosynthesis.

    Uses Eqn 31, after Wullschleger.

    .. math::

        J_{max25} = 29.1 + 1.64 \cdot V_{max25}


    Args:
        Vmax25 (NDArray): The ``Vmax25`` parameter.

    Returns:
        NDArray: The calculated ``Jmax25`` values.
    """

    Jmax25 = 29.1 + 1.64 * Vmax25

    return Jmax25


def Jmax25_temp_correction(Jmax25: NDArray, tc: NDArray) -> NDArray:
    r"""Corrects Jmax value to ambient temperature.

    This function adjusts the maximum electron transport rate ``Jmax25`` for temperature
    using a temperature correction formula, similar to the Arrhenius equation.

    Correction derived from Mengoli 2021 Eqn 3b.

    .. math::

        J_{max} = J_{max25} \cdot \exp\left(\frac{43990}{8.314}
        \left( \frac{1}{298} - \frac{1}{\text{tc} + 273} \right)\right)


    Args:
        Jmax25 (NDArray): The ``Jmax25`` parameter.
        tc (NDArray): The temperature in Celsius.

    Returns:
        NDArray: The temperature-corrected Jmax values.
    """

    Jmax = Jmax25 * np.exp((43990 / 8.314) * (1 / 298 - 1 / (tc + 273)))

    return Jmax


def electron_transport_rate(Jmax: NDArray, I_c: NDArray) -> NDArray:
    r"""Calculate electron transport rate :math:`J`.

    This function calculates the electron transport rate :math:`J`,considering the
    irradiance :math:`I_c` and the maximum electron transport rate :math:`Jmax`.

    .. math::

        J = J_{max} \cdot I_c \cdot \frac{(1 - 0.15)}{(I_c + 2.2 \cdot J_{max})}

    Args:
        Jmax (NDArray): maximum rate of electron transport.
        I_c (NDArray): The irradiance parameter.

    Returns:
        NDArray: The calculated J values.
    """

    J = Jmax * I_c * (1 - 0.15) / (I_c + 2.2 * Jmax)

    return J


def assimilation_rate(mj: NDArray, J: NDArray) -> NDArray:
    r"""Calculate assimilation rate :math:`A`.

    This function calculates the assimilation rate driven by electron transport,
    :math:`Aj`, using the parameter :math:`mj` and the electron transport rate
    :math:`J`.

    .. math::

        A = m_j \cdot \frac{J}{4}

    Args:
        mj (NDArray): The ``mj`` parameter.
        J (NDArray): The ``J`` parameter.

    Returns:
        NDArray: The calculated Aj values.
    """
    A = mj * J / 4

    return A


def assimilation_canopy(
    Aj: NDArray, Av: NDArray, beta_angle: NDArray, solar_obscurity_angle: float
) -> NDArray:
    r"""Calculate assimilation in canopy, :math:`Acanopy`.

    This function calculates the total canopy assimilation by taking the minimum of 
    :math:`Aj` and :math:`Av` for each partition and clipping the data when the sun is 
    below the solar obscurity angle.

    .. math::

        ew_{minima} = \min(A_{j}, A_{v}) \\
        
        A_{canopy} = \begin{cases} 
        0 & \text{if } \beta_{\text{angle}} < \text{solar_obscurity_angle} \\ 
        ew_{minima} & \text{otherwise} 
        \end{cases}

    Args:
        Aj (NDArray): The ``Aj`` parameter.
        Av (NDArray): The ``Av`` parameter.
        beta_angle (NDArray): The solar elevation :math:`beta` angle.
        solar_obscurity_angle (float): The solar obscurity angle.

    Returns:
        NDArray: The calculated assimilation values for the canopy.
    """

    ew_minima = np.minimum(Aj, Av)
    Acanopy = np.where(beta_angle < solar_obscurity_angle, 0, ew_minima)

    return Acanopy


def gross_primary_product(
    k_c_molmass: float, Acanopy_sun: NDArray, Acanopy_shade: NDArray
) -> NDArray:
    r"""Calculate gross primary productivity (``GPP``).

    This function calculates the ``GPP`` by combining the assimilation rates of sunlit
    (:math:`Acanopy_{sun}`) and shaded (:math:`Acanopy_{shade}`) portions of the canopy,
    scaled by a molar mass constant, :math:`k_{c_molmass}`.

    .. math::

        \text{gpp} = k_{\text{c_molmass}} \cdot A_{\text{canopy_sun}} +
        A_{\text{canopy\_shade}}

    Args:
        k_c_molmass (float): The constant for molar mass.
        Acanopy_sun (NDArray): The assimilation values for the sunlit canopy.
        Acanopy_shade (NDArray): The assimilation values for the shaded canopy.

    Returns:
        NDArray: The calculated GPP values.
    """

    gpp = k_c_molmass * Acanopy_sun + Acanopy_shade

    return gpp
