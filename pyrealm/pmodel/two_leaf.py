"""This module implements a version of the two leaf, two stream model of assimilation
:cite:p:`depury:1997a`. The implementation is intended to align closely with the similar
implementation in the BESS model.

The module provides two core classes:

* The :class:`~pyrealm.pmodel.two_leaf.TwoLeafIrradiance` class can be used to estimate
  the irradiance absorbed by sunlit and shaded leaves, given the solar elevation angle,
  the atmospheric pressure, the leaf area index and the photosynthetic photon flux
  density.

* The :class:`~pyrealm.pmodel.two_leaf.TwoLeafAssimilation` then takes an instance of
  the irradiance class and a fitted  :class:`~pyrealm.pmodel.pmodel.PModel` or
  :class:`~pyrealm.pmodel.pmodel.SubdailyPModel`. The carboxylation rate and optimal chi
  calculations from the P Model are then used to calculate the total assimilation by
  sunlit and shaded leaves.

The irradiance calculations are independent of the subsequent calculation of
assimilation, so these classes are separated to allow the same irradiance estimates to
be reused with different assimilation models.
"""  # noqa D210, D415

from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants.core_const import CoreConst
from pyrealm.constants.two_leaf import TwoLeafConst
from pyrealm.core.bounds import BoundsChecker
from pyrealm.core.experimental import warn_experimental
from pyrealm.core.utilities import check_input_shapes
from pyrealm.pmodel.pmodel import PModel, SubdailyPModel

# ------------------------------
# Irradiance class and functions
# ------------------------------


class TwoLeafIrradiance:
    r"""Calculate the irradiance absorbed by sunlit and shaded leaves.

    The two leaf, two stream model of :cite:t:`depury:1997a` partitions the irradiance
    absorbed by the canopy into the irradiance absorbed by sunlit (:math:`I_{csun}`) and
    shaded (:math:`I_{cshade}`) leaves.

    These irradiances are calculated using the canopy leaf area index (:math:`L`) and
    then the solar elevation (:math:`\beta`), photosynthetic photon flux density (PPFD)
    and atmospheric pressure (:math:`P`) as follows:

    * The fraction of diffuse light (:math:`f_d`) is calculated (see
      :meth:`calculate_fraction_of_diffuse_radiation`) and used to partition the
      incoming PPFD into the beam (:math:`I_b = \textrm{PPFD}(1-f_d)`) and diffuse
      (:math:`I_d=PPFD f_d`) irradiances reaching the canopy.

    * Extinction coefficients are calculated for both beam (:math:`k_b`) and scattered
      light (:math:`k_b'`) reaching the canopy, given the solar elevation :math:`\beta`
      (see :meth:`calculate_beam_extinction_coef`).

    * Canopy reflectance coefficients are calculated for both beam and diffuse light.
      The beam reflectance (:math:`\rho_{cb}`) varies with solar elevation through the
      beam extinction coefficient (:math:`k_b`) but diffuse reflectance
      (:math:`\rho_{cd}`) is a constant property of the canopy
      (:attr:`~pyrealm.constants.two_leaf.TwoLeafConst.diffuse_reflectance`).

    * The irradiances for sunlit leaves are then calculated, including components from
      direct beam (:math:`I_{sb}`, see :meth:`calculate_sunlit_beam_irradiance`),
      scattered (:math:`I_{ss}`, see :meth:`calculate_sunlit_scattered_irradiance`) and
      diffuse (:math:`I_{sd}`, see :meth:`calculate_sunlit_diffuse_irradiance`) light.
      The total radiation absorbed by the sunlit leaves (:math:`I_csun`) is the sum of
      these three components.

    * The irradiance for shaded leaves (:math:`I_cshade`) is then calculated as the
      difference between the total canopy irradiance (:math:`I_c`) and the sunlit
      absorbed irradiance (:math:`I_csun`).

    Args:
        solar_elevation: Array of solar elevation angles (radians).
        ppfd: Array of photosynthetic photon flux density values (µmol m-2 s-1).
        leaf_area_index: Array of leaf area index values.
        patm: Array of atmospheric pressure values (pascals).
        core_constants: An instance of the core constants class.
        two_leaf_constants: An instance of the two leaf constants class.
        bounds_checker: A bounds checker instance used to validate the input data.
    """

    __experimental__ = True

    def __init__(
        self,
        solar_elevation: NDArray[np.float64],
        ppfd: NDArray[np.float64],
        leaf_area_index: NDArray[np.float64],
        patm: NDArray[np.float64],
        core_constants: CoreConst = CoreConst(),
        two_leaf_constants: TwoLeafConst = TwoLeafConst(),
        bounds_checker: BoundsChecker = BoundsChecker(),
    ):
        warn_experimental("TwoLeafIrradiance")

        # Check shapes are consistent
        check_input_shapes(solar_elevation, ppfd, leaf_area_index, patm)

        # Check bounds and store input variables
        self.solar_elevation: NDArray[np.float64] = bounds_checker.check(
            "solar_elevation", solar_elevation
        )
        r"""The solar elevation inputs (:math:`\beta`, radians)"""
        self.ppfd: NDArray[np.float64] = bounds_checker.check("ppfd", ppfd)
        """The photosynthetic photon flux density inputs (PPFD, µmol m-2 s-1)"""
        self.leaf_area_index: NDArray[np.float64] = bounds_checker.check(
            "leaf_area_index", leaf_area_index
        )
        """The leaf area index inputs (:math:`L`, unitless)"""
        self.patm: NDArray[np.float64] = bounds_checker.check("patm", patm)
        """The atmospheric pressure (:math:`P`, pascals)"""
        self.core_constants: CoreConst = core_constants
        """An instance of the core constants class."""
        self.two_leaf_constants: TwoLeafConst = two_leaf_constants
        """An instance of the two leaf constants class."""

        # Define instance attributes

        self.fraction_of_diffuse_radiation: NDArray[np.float64]
        """The fraction of diffuse radiation (:math:`f_d`)"""
        self.diffuse_irradiance: NDArray[np.float64]
        """The diffuse irradiance (:math:`I_d`) reaching the canopy."""
        self.beam_irradiance: NDArray[np.float64]
        """The beam irradiance (:math:`I_b`) reaching the canopy."""

        self.beam_extinction_coef: NDArray[np.float64]
        """The beam extinction coefficient (:math:`k_{b}`)"""
        self.scattered_beam_extinction_coef: NDArray[np.float64]
        """The scattered beam extinction coefficient (:math:`k_{b}'`)"""

        self.beam_reflectance: NDArray[np.float64]
        r"""The canopy beam reflectance for leaves with uniform angle distribution
         (:math:`\rho_{cb}`)"""

        self.sunlit_beam_irradiance: NDArray[np.float64]
        """The sunlit beam irradiance (:math:`I_{sb}`)."""
        self.sunlit_diffuse_irradiance: NDArray[np.float64]
        """The sunlit diffuse irradiance (:math:`I_{sd}`)."""
        self.sunlit_scattered_irradiance: NDArray[np.float64]
        """The sunlit scattered irradiance (:math:`I_{ss}`)."""

        self.canopy_irradiance: NDArray[np.float64]
        """The total canopy irradiance (:math:`I_c`)."""

        self.sunlit_absorbed_irradiance: NDArray[np.float64]
        """The sunlit leaf absorbed irradiance (:math:`I_{csun}`)."""
        self.shaded_absorbed_irradiance: NDArray[np.float64]
        """The shaded leaf absorbed irradiance (:math:`I_{cshade}`)."""

        # Automatically calculate the absorbed irradiances.
        self._calculate_absorbed_irradiances()

    def _calculate_absorbed_irradiances(self) -> None:
        r"""Calculate absorbed irradiance for sunlit and shaded leaves."""

        # The fraction of diffuse radiation: f_d
        self.fraction_of_diffuse_radiation = calculate_fraction_of_diffuse_radiation(
            patm=self.patm,
            solar_elevation=self.solar_elevation,
            standard_pressure=self.core_constants.k_Po,
            atmospheric_scattering=self.two_leaf_constants.atmospheric_scattering_coef,
            atmos_transmission_par=self.two_leaf_constants.atmos_transmission_par,
        )

        # The diffuse and beam irradiance reaching the canopy: I_b, I_d
        self.beam_irradiance = (1 - self.fraction_of_diffuse_radiation) * self.ppfd
        self.diffuse_irradiance = self.fraction_of_diffuse_radiation * self.ppfd

        # The beam extinction coefficient for direct light: k_b
        self.beam_extinction_coef = calculate_beam_extinction_coef(
            solar_elevation=self.solar_elevation,
            solar_obscurity_angle=self.two_leaf_constants.solar_obscurity_angle,
            extinction_numerator=self.two_leaf_constants.direct_beam_extinction_numerator,
        )

        # The extinction coefficient for scattered light: k_b'
        self.scattered_beam_extinction_coef = calculate_beam_extinction_coef(
            solar_elevation=self.solar_elevation,
            solar_obscurity_angle=self.two_leaf_constants.solar_obscurity_angle,
            extinction_numerator=self.two_leaf_constants.scattered_beam_extinction_numerator,
        )

        # The canopy beam reflectance for leaves with a uniform angle distribution:
        # rho_cb
        self.beam_reflectance = calculate_beam_reflectance(
            beam_extinction=self.beam_extinction_coef,
            horizontal_leaf_reflectance=self.two_leaf_constants.horizontal_leaf_reflectance,
        )

        # Calculate fractions of the sunlit leaf irradiance
        self.sunlit_beam_irradiance = calculate_sunlit_beam_irradiance(
            beam_irradiance=self.beam_irradiance,
            beam_extinction_coef=self.beam_extinction_coef,
            leaf_area_index=self.leaf_area_index,
            leaf_scattering_coef=self.two_leaf_constants.leaf_scattering_coef,
        )

        self.sunlit_diffuse_irradiance = calculate_sunlit_diffuse_irradiance(
            diffuse_irradiance=self.diffuse_irradiance,
            beam_extinction_coef=self.beam_extinction_coef,
            leaf_area_index=self.leaf_area_index,
            diffuse_reflectance=self.two_leaf_constants.diffuse_reflectance,
            diffuse_extinction_coef=self.two_leaf_constants.diffuse_extinction_coef,
        )

        self.sunlit_scattered_irradiance = calculate_sunlit_scattered_irradiance(
            beam_irradiance=self.beam_irradiance,
            beam_reflectance=self.beam_reflectance,
            scattered_beam_extinction_coef=self.scattered_beam_extinction_coef,
            beam_extinction_coef=self.beam_extinction_coef,
            leaf_area_index=self.leaf_area_index,
            leaf_scattering_coef=self.two_leaf_constants.leaf_scattering_coef,
        )

        # And hence the total sunlit irradiance as the sum of those three components
        self.sunlit_absorbed_irradiance = (
            self.sunlit_beam_irradiance
            + self.sunlit_diffuse_irradiance
            + self.sunlit_scattered_irradiance
        )

        # Now calculate the irradiance absorbed by shaded leaves
        # Calculate canopy irradiance
        self.canopy_irradiance = calculate_canopy_irradiance(
            beam_reflectance=self.beam_reflectance,
            beam_irradiance=self.beam_irradiance,
            scattered_beam_extinction_coef=self.scattered_beam_extinction_coef,
            diffuse_radiation=self.diffuse_irradiance,
            leaf_area_index=self.leaf_area_index,
            diffuse_reflectance=self.two_leaf_constants.diffuse_reflectance,
            diffuse_extinction_coef=self.two_leaf_constants.diffuse_extinction_coef,
        )

        self.shaded_absorbed_irradiance = np.where(
            self.solar_elevation > self.two_leaf_constants.solar_obscurity_angle,
            self.canopy_irradiance - self.sunlit_absorbed_irradiance,
            0,
        )


def calculate_beam_extinction_coef(
    solar_elevation: NDArray[np.float64],
    solar_obscurity_angle: float = TwoLeafConst().solar_obscurity_angle,
    extinction_numerator: float = TwoLeafConst().direct_beam_extinction_numerator,
) -> NDArray[np.float64]:
    r"""Calculate the beam extinction coefficient.

    The beam extinction coefficient (:math:`k_b`) captures changes in the the
    attenuation of direct sunlight through the canopy with variation in the solar
    elevation angle.

    .. math::

        k_b = \frac{n}{\sin\left(\max\left(\beta, \beta_{ob} \right)\right)}

    Args:
        solar_elevation: Array of solar elevation angles (:math:`\beta`)
        solar_obscurity_angle: Threshold angle at which the beam extinction coefficient
            reaches a maximum (:math:`\beta_{ob}`).
        extinction_numerator: The numerator used in the calculation (:math:`n`).

    Returns:
        An array of beam extinction coefficients.
    """

    return np.where(
        solar_elevation > solar_obscurity_angle,
        extinction_numerator / np.sin(solar_elevation),
        extinction_numerator / np.sin(solar_obscurity_angle),
    )


def calculate_fraction_of_diffuse_radiation(
    patm: NDArray[np.float64],
    solar_elevation: NDArray[np.float64],
    standard_pressure: float = CoreConst().k_Po,
    atmospheric_scattering: float = TwoLeafConst().atmospheric_scattering_coef,
    atmos_transmission_par: float = TwoLeafConst().atmos_transmission_par,
) -> NDArray[np.float64]:
    r"""Calculate the fraction of diffuse radiation.

    The fraction of diffuse radiation (:math:`f_d`) captures the proportion of
    sunlight that is scattered before reaching the canopy, following equation A23 and
    A25 of :cite:t:`depury:1997a`.

    .. math::
        :nowrap:

        \[
            \begin{align*}
                m &= \left(P / P_0 \right) / \sin \beta \\
                f_d &= \frac{1 - a^m}{1 + a^m \left(1 / f_a - 1\right)}
            \end{align*}
        \]


    Args:
        patm: Atmospheric pressure values (:math:`P`, pascals).
        solar_elevation: Solar elevation angles (:math:\beta`, radians).
        standard_pressure: Standard atmospheric pressure (:math:`P_0`, pascals).
        atmospheric_scattering: Atmospheric scattering factor (:math:`f_a`).
        atmos_transmission_par:  The atmospheric transmission coefficient of
            photosynthetically active radiation (:math:`a`).

    Returns:
        Array of fractions of diffuse radiation.
    """

    # Optical air mass
    m = (patm / standard_pressure) / np.sin(solar_elevation)

    # Diffuse fraction
    f_d = (1 - atmos_transmission_par**m) / (
        1 + (atmos_transmission_par**m * (1 / atmospheric_scattering - 1))
    )

    # Test for negative values - it isn't clear that this actually occurs but the
    # reference implementation clipped the resulting irradiances at zero, so this is
    # here to safeguard and inform.
    if np.any(f_d < 0):
        f_d = np.clip(a=f_d, a_min=0, a_max=None)
        warn("Negative diffuse radiation fractions clamped to zero.")

    return f_d


def calculate_beam_reflectance(
    beam_extinction: NDArray[np.float64],
    horizontal_leaf_reflectance: float = TwoLeafConst().horizontal_leaf_reflectance,
) -> NDArray[np.float64]:
    r"""Calculate the beam irradiance for leaves with a uniform angle distribution.

    The beam irradiance with a uniform leaf angle distribution (:math:`\rho_{cb}`)
    captures different leaf orientations within the canopy, following equation A19 of
    :cite:t:`depury:1997a`

    .. math::

        \rho_{cb} = 1 - \exp \left(-\frac{2 \rho_h \, k_b}{1 + k_b}\right)

    Args:
        beam_extinction: Array of beam extinction coefficients (:math:`k_b`).
        horizontal_leaf_reflectance: The reflectance coefficient horizontal leaves
            (:math:`\rho_h`).

    Returns:
        Array of beam irradiances.
    """

    return 1.0 - np.exp(
        -2 * horizontal_leaf_reflectance * beam_extinction / (1 + beam_extinction)
    )


def calculate_canopy_irradiance(
    beam_reflectance: NDArray[np.float64],
    beam_irradiance: NDArray[np.float64],
    scattered_beam_extinction_coef: NDArray[np.float64],
    diffuse_radiation: NDArray[np.float64],
    leaf_area_index: NDArray[np.float64],
    diffuse_reflectance: float = TwoLeafConst().diffuse_reflectance,
    diffuse_extinction_coef: float = TwoLeafConst().diffuse_extinction_coef,
) -> NDArray[np.float64]:
    r"""Calculate the canopy irradiance.

    The canopy irradiance (:math:`I_c`) is the total irradiance within the canopy,
    including both direct and diffuse radiation components.

    .. math::

        I_c = (1 - \rho_{cb})  I_b   (1 - \exp(-k_b' L)) +
            (1 - \rho_{cd})  I_d  (1 - \exp(-k_d' L))

    Args:
        beam_reflectance : The beam reflectance of leaves with uniform angle
            distribution  (:math:`\rho_{cb}`).
        beam_irradiance : Beam irradiance values (:math:`I_b`).
        scattered_beam_extinction_coef : Scattered beam extinction
            coefficients (:math:`k_b'`).
        diffuse_radiation : Diffuse radiation values (:math:`I_d`).
        leaf_area_index : The leaf area index  (:math:`L`).
        diffuse_reflectance : The canopy reflectance of diffuse radiation
            (:math:`\rho_{cd}`).
        diffuse_extinction_coef : The diffuse light extinction coefficients
            (:math:`k_d'`).

    Returns:
        Canopy irradiance values.
    """

    return (1 - beam_reflectance) * beam_irradiance * (
        1 - np.exp(-scattered_beam_extinction_coef * leaf_area_index)
    ) + (1 - diffuse_reflectance) * diffuse_radiation * (
        1 - np.exp(-diffuse_extinction_coef * leaf_area_index)
    )


def calculate_sunlit_beam_irradiance(
    beam_irradiance: NDArray[np.float64],
    beam_extinction_coef: NDArray[np.float64],
    leaf_area_index: NDArray[np.float64],
    leaf_scattering_coef: float = TwoLeafConst().leaf_scattering_coef,
) -> NDArray[np.float64]:
    r"""Calculate the sunlit beam irradiance.

    The sunlit beam irradiance (:math:`I_{sun_beam}`) is the direct sunlight received by
    the sunlit portion of the canopy.

    .. math::

        I_{sun\_beam} = I_b (1 - \sigma) (1 - \exp{-k_b L})

    Args:
        beam_irradiance: The irradiance by direct beam light (:math:`I_b`)
        beam_extinction_coef: Beam extinction coefficients (:math:`k_b`)
        leaf_area_index: The leaf area index for the canopy (:math:`L`)
        leaf_scattering_coef: The scattering coefficient of PAR by leaves
            (:math:`\sigma`).

    Returns:
        Array of sunlit beam irradiance values.
    """
    return (
        beam_irradiance
        * (1 - leaf_scattering_coef)
        * (1 - np.exp(-beam_extinction_coef * leaf_area_index))
    )


def calculate_sunlit_diffuse_irradiance(
    diffuse_irradiance: NDArray[np.float64],
    beam_extinction_coef: NDArray[np.float64],
    leaf_area_index: NDArray[np.float64],
    diffuse_reflectance: float = TwoLeafConst().diffuse_reflectance,
    diffuse_extinction_coef: float = TwoLeafConst().diffuse_extinction_coef,
) -> NDArray[np.float64]:
    r"""Calculate the sunlit diffuse irradiance.

    The sunlit diffuse irradiance (:math:`I_{s_d}`) is the diffuse radiation
    received by the sunlit portion of the canopy.

    .. math::

        I_{sd} = I_d  (1 - \rho_{cd}) (1 - \exp(-(k_d' + k_b)
            L \frac{k_d'}{k_d' + k_b}

    Args:
        diffuse_irradiance: Array of diffuse radiation values (:math:`I_d`)
        beam_extinction_coef: Array of beam extinction coefficients (:math:`k_b`)
        leaf_area_index: Array of leaf area index values (:math:`L`)
        diffuse_reflectance : The canopy reflectance of diffuse radiation
            (:math:`\rho_{cd}`).
        diffuse_extinction_coef: Constant for calculating the sunlit diffuse
            irradiance (:math:`k_d'`)

    Returns:
        Array of sunlit diffuse irradiance values.
    """
    return (
        diffuse_irradiance
        * (1 - diffuse_reflectance)
        * (
            1
            - np.exp(
                -(diffuse_extinction_coef + beam_extinction_coef) * leaf_area_index
            )
        )
        * diffuse_extinction_coef
        / (diffuse_extinction_coef + beam_extinction_coef)
    )


def calculate_sunlit_scattered_irradiance(
    beam_irradiance: NDArray[np.float64],
    beam_reflectance: NDArray[np.float64],
    scattered_beam_extinction_coef: NDArray[np.float64],
    beam_extinction_coef: NDArray[np.float64],
    leaf_area_index: NDArray[np.float64],
    leaf_scattering_coef: float = TwoLeafConst().leaf_scattering_coef,
) -> NDArray[np.float64]:
    r"""Calculate the sunlit scattered irradiance.

    The sunlit scattered irradiance (:math:`I_{ss}`) is the scattered
    radiation received by the sunlit portion of the canopy.

    .. math::

        I_{ss} = I_b  ((1 - \rho_{cb}) (1 - \exp(-(k_b' + k_b) L))
            \frac{k_b'}{k_b' + k_b} - (1 - \sigma) (1 - \exp(-2 k_b  L)) / 2)

    Args:
        beam_irradiance: Array of beam irradiance values (:math:`I_b`)
        beam_reflectance : The beam reflectance of leaves with uniform angle
            distribution  (:math:`\rho_{cb}`).
        scattered_beam_extinction_coef: Array of scattered beam extinction
            coefficients (:math:`k_b'`)
        beam_extinction_coef: Array of beam extinction coefficients (:math:`k_b`)
        leaf_area_index: Array of leaf area index values (:math:`L`)
        leaf_scattering_coef: The scattering coefficient of PAR by leaves
            (:math:`\sigma`).

    Returns:
        Array of sunlit scattered irradiance values.
    """

    return beam_irradiance * (
        (1 - beam_reflectance)
        * (
            1
            - np.exp(
                -(scattered_beam_extinction_coef + beam_extinction_coef)
                * leaf_area_index
            )
        )
        * scattered_beam_extinction_coef
        / (scattered_beam_extinction_coef + beam_extinction_coef)
        - (1 - leaf_scattering_coef)
        * (1 - np.exp(-2 * beam_extinction_coef * leaf_area_index))
        / 2
    )


# ------------------------------
# Assimilation class and functions
# ------------------------------


class TwoLeafAssimilation:
    r"""Estimate gross primary production using the two-leaf, two-stream model.

    The two leaf, two stream model of :cite:t:`depury:1997a` estimates irradiance within
    the canopy from diffuse, scattered and direct beam light radiation and then
    separately estimates the assimilation by sunlit and shaded leaves.

    The class requires an estimate of the irradiances absorbed by the sunlit and shaded
    leaves, calculated using the :class:`~pyrealm.pmodel.two_leaf.TwoLeafIrradiance`
    class. It also requires a :class:`~pyrealm.pmodel.pmodel.PModel` or
    :class:`~pyrealm.pmodel.pmodel.SubdailyPModel` instance, which is used to provide
    estimates of four parameters:

    * the maximum carboxylation rate (:math:`V_{cmax}`),
    * the maximum carboxylation rate at standard temperature (:math:`V_{cmax25}`),
    * the :math:`\ce{CO2}` limitation factor for RuBisCO-limited assimilation
      (:math:`m_c`), and
    * the :math:`\ce{CO2}` limitation factor for light-limited assimilation
      (:math:`m_j`).

    The model then calculate assimilation as follows:

    * An extinction coefficient (:math:`k_v`) is calculated to account for changes in
      :math:`V_{cmax}` with depth in the canopy, following an empirical function in
      :cite:t:`lloyd:2010a` (see :meth:`calculate_canopy_extinction_coef`) and using
      leaf area index (:math:`L`) as a proxy for canopy depth.

    * The value for :math:`V_{cmax25}` from the P Model is adjusted to give a
      representative through the canopy (:math:`V_{cmax25\_C}`) using :math:`L` and
      :math:`k_v` (see :meth:`calculate_canopy_vcmax25`). This carboxylation capacity is
      then partitioned between sunlit and shaded leaves: :math:`V_{cmax25\_Sn}` is
      estimated using :meth:`calculate_sun_vcmax25` and then :math:`V_{cmax25\_Sd} =
      V_{cmax25\_C} - V_{cmax25\_Sn}`.

    * Values for :math:`J_{max25}` are then calculated using an empirical function of
      :math:`V_{cmax25`}` for both sunlit (:math:`J_{max25\_Sn}`) and shaded
      (:math:`J_{max25\_Sd}`) leaves (see :meth:`calculate_jmax25`).

    * The Arrhenius scaling method used with the P Model is then used to adjust these
      estimates to the observed temperatures, giving :math:`V_{cmax\_Sn}, V_{cmax\_Sd},
      J_{max\_Sn}, J_{max\_Sd}`.

    * The maximum assimilation rate via the carboxylation pathway is then found as
      :math:`A_v = V_{cmax} m_c` for each leaf type, giving :math:`A_{v\_Sn}` and
      :math:`A_{v\_Sd}`.

    * The realised electron transport rate :math:`J` is calculated using an
      epmirical function (see :meth:`calculate_electron_transport_rate`) and then used
      to calculate the maximum assimilation rate via the electron transfer pathway as
      :math:`A_j = m_j J / 4`, giving  :math:`A_{j\_Sn}` and :math:`A_{j\_Sd}`.

    * The realised assimilation is the minimum of  :math:`A_{c}` and :math:`A_{j}` for
      each leaf type, giving:

        * :math:`A_{Sn} = \min \left( A_{v\_Sn}, A_{j\_Sn} \right)`,
        * :math:`A_{Sd} = \min \left( A_{v\_Sd}, A_{j\_Sd} \right)`.

    *  The gross primary productivity (gC m-2 s-1) is then calculated as :math:`(A_{Sn}
       + A_{Sd}) * M_C`, where :math:`M_C` is the molar mass of carbon.

    Args:
        pmodel: A PModel or SubdailyPModel.
        irrad: An instance of TwoLeafIrradiance.
    """

    __experimental__ = True

    def __init__(
        self,
        pmodel: PModel | SubdailyPModel,
        irradiance: TwoLeafIrradiance,
    ):
        """Initialize the TwoLeafAssimilation class."""

        warn_experimental("TwoLeafAssimilation")

        self.pmodel = pmodel
        """A PModel or SubdailyPModel instance."""
        self.irradiance = irradiance
        """A TwoLeafIrradiance instance """

        # TODO - both the pmodel and irrad instances have their own independent
        #        CoreConst instances. Would be good to check they are the same. Because
        #        we want to able to use the irrad object with multiple models, it would
        #        be convoluted to force them to be the same instance. We'd need to write
        #        a custom __eq__ dunder method to handle the structures inside the
        #        classes.
        #      - Similarly, both the PModel and TwoLeafIrradiance require PPFD and PATM
        #        and those should be the same between the two objects.
        #      - Ideally we want to avoid these dual definitions, which we could do if
        #        the PModel is used as input to the TwoLeafIrradiance. Less reusable
        #        class, but removes the parallel arguments.
        #      - See https://github.com/ImperialCollegeLondon/pyrealm/issues/469

        self.canopy_extinction_coef: NDArray[np.float64]
        """An extinction coefficient capturing the vertical structure of carboxylation
        capacity within the canopy (:math:`k_v`)."""
        self.Vcmax25_canopy: NDArray[np.float64]
        r"""The total canopy carboxylation capacity at standard temperature
        :math:`V_{cmax25\_C}`"""
        self.Vcmax25_sun: NDArray[np.float64]
        r"""The maximum rate of carboxylation at standard temperature within sunlit 
        leaves (:math:`V_{cmax25\_Sn}`)"""
        self.Vcmax25_shade: NDArray[np.float64]
        r"""The maximum rate of carboxylation at standard temperature within shaded 
        leaves (:math:`V_{cmax25\_Sd}`)"""
        self.Vcmax_sun: NDArray[np.float64]
        r"""The maximum rate of carboxylation at the observed temperature within sunlit
        leaves (:math:`V_{cmax\_Sn}`)"""
        self.Vcmax_shade: NDArray[np.float64]
        r"""The maximum rate of carboxylation at the observed temperature within shaded
        leaves (:math:`V_{cmax\_Sd}`)"""
        self.Jmax25_sun: NDArray[np.float64]
        r"""The maximum rate of electron transfer at standard temperature within sunlit
        leaves (:math:`J_{max25\_Sn}`)"""
        self.Jmax25_shade: NDArray[np.float64]
        r"""The maximum rate of electron transfer at standard temperature within shaded
        leaves (:math:`J_{max25\_Sd}`)"""
        self.Jmax_sun: NDArray[np.float64]
        r"""The maximum rate of electron transfer at the observed temperature within
        sunlit leaves (:math:`J_{max\_Sn}`)"""
        self.Jmax_shade: NDArray[np.float64]
        r"""The maximum rate of electron transfer at the observed temperature within
        shaded leaves (:math:`J_{max25\_Sn}`)"""
        self.J_sun: NDArray[np.float64]
        r"""The realised rate of electron transfer within sunlit leaves
        (:math:`J_{Sn}`"""
        self.J_shade: NDArray[np.float64]
        r"""The realised rate of electron transfer within sunlit leaves
        (:math:`J_{Sd}`"""
        self.Av_sun: NDArray[np.float64]
        r"""The potential rate of assimilation associated with carboxylation in sunlit
        leaves (:math:`A_{v\_Sn}`)."""
        self.Av_shade: NDArray[np.float64]
        r"""The potential rate of assimilation associated with carboxylation in shaded
        leaves (:math:`A_{v\_Sd}`)."""
        self.Aj_sun: NDArray[np.float64]
        r"""The potential rate of assimilation associated with electron transfer in
        sunlit leaves (:math:`A_{j\_Sn}`)."""
        self.Aj_shade: NDArray[np.float64]
        r"""The potential rate of assimilation associated with electron transfer in
        shaded leaves (:math:`A_{j\_Sn}`)."""
        self.A_sun: NDArray[np.float64]
        r"""The realised assimilation rate for sunlit leaves (:math:`A_{Sn}`)."""
        self.A_shade: NDArray[np.float64]
        r"""The realised assimilation rate for shaded leaves (:math:`A_{Sd}`)."""
        self.gpp: NDArray[np.float64]
        r"""The gross primary productivity across the sunlit and shaded leaves."""

        # Automatically run the
        self._calculate_two_leaf_two_stream_gpp()

    def _calculate_two_leaf_two_stream_gpp(self) -> None:
        """Internal calculations for the two leaf, two stream model."""

        # Calculate the canopy extinction coefficient given the big leaf estimate of
        # V_cmax
        self.canopy_extinction_coef = calculate_canopy_extinction_coef(
            vcmax=self.pmodel.vcmax,
            coef=self.irradiance.two_leaf_constants.vcmax_lloyd_coef,
        )

        # Calculate overall Vcmax25 for the canopy and then partition between sunlit and
        # shaded leaves
        self.Vcmax25_canopy = calculate_canopy_vcmax25(
            leaf_area_index=self.irradiance.leaf_area_index,
            vcmax25=self.pmodel.vcmax25,
            canopy_extinction_coef=self.canopy_extinction_coef,
        )

        self.Vcmax25_sun = calculate_sun_vcmax25(
            leaf_area_index=self.irradiance.leaf_area_index,
            vcmax25=self.pmodel.vcmax25,
            canopy_extinction_coef=self.canopy_extinction_coef,
            beam_extinction_coef=self.irradiance.beam_extinction_coef,
        )

        self.Vcmax25_shade = self.Vcmax25_canopy - self.Vcmax25_sun

        # Calculate Jmax25 for sunlit and shaded leaves as an empirical function of
        # Vcmax25
        self.Jmax25_sun = calculate_jmax25(self.Vcmax25_sun)
        self.Jmax25_shade = calculate_jmax25(self.Vcmax25_shade)

        # Set up Arrhenius scaling to get rates at environmental temperatures using the
        # Arrhenius method selected in the original P Model.
        arrhenius_factors = self.pmodel._arrhenius_class(env=self.pmodel.env)

        # Scale Vcmax25 to Vcmax and Jmax25 to Jmax for sunlit and shaded leaves
        arrhenius_vcmax = arrhenius_factors.calculate_arrhenius_factor(
            coefficients=self.pmodel.pmodel_const.arrhenius_vcmax
        )
        self.Vcmax_sun = self.Vcmax25_sun * arrhenius_vcmax
        self.Vcmax_shade = self.Vcmax25_shade * arrhenius_vcmax

        arrhenius_jmax = arrhenius_factors.calculate_arrhenius_factor(
            coefficients=self.pmodel.pmodel_const.arrhenius_jmax
        )
        self.Jmax_sun = self.Jmax25_sun * arrhenius_jmax
        self.Jmax_shade = self.Jmax25_shade * arrhenius_jmax

        # Calculate Av and Aj for both sun and shaded leaves
        self.Av_sun = self.Vcmax_sun * self.pmodel.optchi.mc
        self.Av_shade = self.Vcmax_shade * self.pmodel.optchi.mc

        self.J_sun = calculate_electron_transport_rate(
            jmax=self.Jmax_sun,
            absorbed_irradiance=self.irradiance.sunlit_absorbed_irradiance,
        )
        self.J_shade = calculate_electron_transport_rate(
            jmax=self.Jmax_shade,
            absorbed_irradiance=self.irradiance.shaded_absorbed_irradiance,
        )

        self.Aj_sun = self.pmodel.optchi.mj * self.J_sun / 4
        self.Aj_shade = self.pmodel.optchi.mj * self.J_shade / 4

        # Calculate the sun and shaded leaf assimilation as the minimum of Ac and Aj for
        # each component
        self.Acanopy_sun = np.minimum(self.Aj_sun, self.Av_sun)
        self.Acanopy_shade = np.minimum(self.Aj_shade, self.Av_shade)

        # Calculate GPP in gC m-2 s-1 as the sum of the sunlit and shaded components,
        # but explicitly setting to zero where the solar elevation is below the solar
        # obscurity angle.
        self.gpp = np.where(
            self.irradiance.solar_elevation
            > self.irradiance.two_leaf_constants.solar_obscurity_angle,
            (self.Acanopy_shade + self.Acanopy_sun)
            * self.irradiance.core_constants.k_c_molmass,
            0,
        )


def calculate_canopy_extinction_coef(
    vcmax: NDArray[np.float64],
    coef: tuple[float, float] = TwoLeafConst().vcmax_lloyd_coef,
) -> NDArray[np.float64]:
    r"""Calculate the canopy extinction coefficient.

    The extinction coefficient (:math:`k_v`) captures the decrease in photosynthetic
    capacity (:math:`V_{cmax}`) with depth in the plant canopy.

    The exponential model used here is taken from Figure 10 of :cite:`lloyd:2010a`,
    which presents an empirical model of the vertical profile in photosynthetic
    capacity, using data from the Amazon forest.

    .. math::
        kv_{Lloyd} = \exp(0.00963 \cdot vcmax_{pmod} - 2.43)

    Args:
        vcmax: The ``vcmax`` attribute from a P Model (:math:`V_{cmax}`).
        coef: The coefficients of the exponential momdel.

    Returns:
        NDArray: The calculated :math:`kv_Lloyd` values.
    """

    a, b = coef
    return np.exp(a * vcmax - b)


def calculate_canopy_vcmax25(
    leaf_area_index: NDArray[np.float64],
    vcmax25: NDArray[np.float64],
    canopy_extinction_coef: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate standardised carboxylation rate in the canopy.

    This function calculates the maximum carboxylation rate of the canopy at a reference
    temperature of 25°C (:math:`V_{cmax25\_C}`), given the depth of the canopy as
    estimated using leaf area index and the rate of decrease in carboxylation rate
    through the canopy, as calculated using
    :meth:`calculate_canopy_extinction_coef`.

    .. math::
        V_{cmax25\_C} = L \, V_{cmax25}  \left(\frac{1 - \exp(-k_v)}{k_v}\right)

    Args:
        leaf_area_index: The leaf area index (:math:`L`).
        vcmax25: The ``vcmax25`` parameter from a P Model (:math:`V_{cmax25}`).
        canopy_extinction_coef: The canopy extinction coefficient (:math:`k_v`).

    Returns:
        NDArray: The calculated Vmax25 canopy values.
    """
    return (
        leaf_area_index
        * vcmax25
        * ((1 - np.exp(-canopy_extinction_coef)) / canopy_extinction_coef)
    )


def calculate_sun_vcmax25(
    leaf_area_index: NDArray[np.float64],
    vcmax25: NDArray[np.float64],
    canopy_extinction_coef: NDArray[np.float64],
    beam_extinction_coef: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate standardised carboxylation rate of sunlit leaves.

    Calculates the maximum carboxylation rate for sunlit leaves at the standard
    temperature of 25°C  (:math:`V_{cmax25\_Sn}`) as:

    .. math::

        V_{cmax25\_Sn} = L \, V_{cmax25} \left(
            \frac{1 - \exp{-k_v - k_b}  L )} {k_v + k_b L}
            \right)

    Args:
        leaf_area_index: The leaf area index (LAI, :math:`L`).
        vcmax25: The ``vcmax25`` parameter from a P Model (:math:`V_{cmax25}`).
        canopy_extinction_coef: The canopy extinction coefficient (:math:`k_v`).
        beam_extinction_coef: The beam extinction coefficient from the irradiance
            model (:math:`k_b`).

    Returns:
        NDArray: The calculated Vmax25 sun values.
    """
    Vmax25_sun = (
        leaf_area_index
        * vcmax25
        * (
            (
                1
                - np.exp(
                    -canopy_extinction_coef - beam_extinction_coef * leaf_area_index
                )
            )
            / (canopy_extinction_coef + beam_extinction_coef * leaf_area_index)
        )
    )

    return Vmax25_sun


def calculate_jmax25(
    vcmax25: NDArray[np.float64],
    coef: tuple[float, float] = TwoLeafConst().jmax25_wullschleger_coef,
) -> NDArray[np.float64]:
    r"""Calculate the maximum rate of electron transport.

    This function calculates the maximum rate of electron transport (:math:`J_{max25}`)
    at 25°C as an linear function of the carboxylation rate (:math:`V_{cmax25}`),
    following the fitted model in Figure 2 of :cite:`wullschleger:1993a`. The default
    values for the coefficients (:math:`a=29,b=1.64`) come from the same source.

    .. math::

        J_{max25} = a + b V_{cmax25}

    Args:
        vcmax25: An estimate of :math:`V_{cmax25}`.
        coef: The coefficients of the empirical relationship between :math:`V_{cmax25}`
            and :math:`J_{max25}`.

    Returns:
        The calculated values of :math:`J_{max25}`.
    """

    a, b = coef
    return a + b * vcmax25


def calculate_electron_transport_rate(
    jmax: NDArray[np.float64], absorbed_irradiance: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Calculate electron transport rate.

    This function calculates the realised electron transport rate (:math:`J`), given
    absorbed irradiance and the maximum electron transport rate :math:`J_{max}`.

    .. math::

        J = J_{max}  I_c \frac{(1 - 0.15)}{(I_c + 2.2  J_{max})}

    Args:
        jmax: The maximum rate of electron transport (:math:`J_{max}`).
        absorbed_irradiance: The abbsorbed irradiance (:math:`I_c`)

    Returns:
        The calculated J values.
    """

    # TODO  What is the source of this parameterisation?

    J = jmax * absorbed_irradiance * (1 - 0.15) / (absorbed_irradiance + 2.2 * jmax)

    return J
