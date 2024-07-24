"""To do."""

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import core_const
from pyrealm.constants.two_leaf_canopy import TwoLeafConst
from pyrealm.pmodel import PModel
from pyrealm.pmodel.optimal_chi import OptimalChiABC
from pyrealm.pmodel.subdaily import SubdailyPModel


class TwoLeafIrradience:
    """Running the two leaf, two stream model within Pyrealm.

    This class implements the methodology of Pury and Farquhar (1997)
    :cite:p:`Pury&Farquhar:1997` two leaf, two stream model. This model is chosen to
    provide a better representation than the big leaf model and to align closely to
    the workings of the BESS model (Ryuet al. 2011):cite:p:`Ryu_et_al:2011`.

    Args:
        beta_angle (NDArray): Array of beta angles (radians).
        ppfd (NDArray): Array of photosynthetic photon flux density values.
        leaf_area_index (NDArray): Array of leaf area index values.
        patm (NDArray): Array of atmospheric pressure values.
    """

    def __init__(
        self,
        beta_angle: NDArray,
        ppfd: NDArray,
        leaf_area_index: NDArray,
        patm: NDArray,
    ):
        self.beta_angle = beta_angle
        self.ppfd = ppfd
        self.leaf_area_index = leaf_area_index
        self.patm = patm

        constants = TwoLeafConst()

        self.pa0: float = constants.k_PA0
        self.k_fa: float = constants.k_fa
        self.k_sigma: float = constants.k_sigma
        self.k_rho_cd: float = constants.k_rho_cd
        self.k_kd_prime: float = constants.k_kd_prime
        self.k_sol_obs_angle: float = constants.k_sol_obs_angle

        self.shapes_agree = self._check_input_consistency()

    def _check_input_consistency(self) -> bool:
        """Check if input arrays have consistent shapes.

        Returns:
            bool: True if all input arrays have the same shape, False otherwise.
        """
        try:
            arrays = [self.beta_angle, self.ppfd, self.leaf_area_index, self.patm]
            shapes = [array.shape for array in arrays]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError("Input arrays have inconsistent shapes.")
            return True
        except Exception as e:
            print(f"Error in input consistency check: {e}")
            return False

    def _calc_beam_extinction_coeff(self) -> NDArray:
        """Calculate the beam extinction coefficient.

        Returns:
            NDArray: Array of beam extinction coefficients.
        """
        kb = np.where(
            self.beta_angle > self.k_sol_obs_angle, 0.5 / np.sin(self.beta_angle), 30
        )
        return kb

    def _calc_scattered_beam_extinction_coeff(self) -> NDArray:
        """Calculate the scattered beam extinction coefficient.

        Returns:
            NDArray: Array of scattered beam extinction coefficients.
        """
        kb_prime = np.where(
            self.beta_angle > self.k_sol_obs_angle, 0.46 / np.sin(self.beta_angle), 27
        )
        return kb_prime

    def _calc_fraction_of_diffuse_rad(self) -> NDArray:
        """Calculate the fraction of diffuse radiation.

        Returns:
            NDArray: Array of fractions of diffuse radiation.
        """
        m = (self.patm / self.pa0) / np.sin(self.beta_angle)
        fd = (1 - 0.72**m) / (1 + (0.72**m * (1 / self.k_fa - 1)))
        return fd

    def _calc_beam_irradience_h_leaves(self) -> NDArray:
        """Calculate the beam irradiance for horizontal leaves.

        Returns:
            NDArray: Array of beam irradiances for horizontal leaves.
        """
        rho_h = (1 - np.sqrt(1 - self.k_sigma)) / (1 + np.sqrt(1 - self.k_sigma))
        return rho_h

    def _calc_beam_irrad_unif_leaf_angle_dist(
        self, rho_h: NDArray, kb: NDArray
    ) -> NDArray:
        """Calculate the beam irradiance with a uniform leaf angle distribution.

        Args:
            rho_h (NDArray): Array of beam irradiances for horizontal leaves.
            kb (NDArray): Array of beam extinction coefficients.

        Returns:
            NDArray: Array of beam irradiances with uniform leaf angle distribution.
        """
        rho_cb = 1 - np.exp(-2 * rho_h * kb / (1 + kb))
        return rho_cb

    def _calc_diffuse_rad(self, fd: NDArray) -> NDArray:
        """Calculate the diffuse radiation.

        Args:
            fd (NDArray): Array of fractions of diffuse radiation.

        Returns:
            NDArray: Array of diffuse radiation values.
        """
        I_d = np.clip(self.ppfd * fd, a_min=0, a_max=np.inf)
        return I_d

    def _calc_beam_irradience(self, fd: NDArray) -> NDArray:
        """Calculate the beam irradiance.

        Args:
            fd (NDArray): Array of fractions of diffuse radiation.

        Returns:
            NDArray: Array of beam irradiance values.
        """
        I_b = self.ppfd * (1 - fd)
        return I_b

    def _calc_scattered_beam_irradience(
        self, I_b: NDArray, kb: NDArray, kb_prime: NDArray, rho_cb: NDArray
    ) -> NDArray:
        """Calculate the scattered beam irradiance.

        Args:
            I_b (NDArray): Array of beam irradiance values.
            kb (NDArray): Array of beam extinction coefficients.
            kb_prime (NDArray): Array of scattered beam extinction coefficients.
            rho_cb (NDArray): Array of beam irradiances with
                uniform leaf angle distribution.

        Returns:
            NDArray: Array of scattered beam irradiance values.
        """
        I_bs = I_b * (1 - rho_cb) * kb_prime * np.exp(
            -kb_prime * self.leaf_area_index
        ) - (1 - self.k_sigma) * kb * np.exp(-kb * self.leaf_area_index)
        return I_bs

    def _calc_canopy_irradience(
        self, rho_cb: NDArray, I_b: NDArray, kb_prime: NDArray, I_d: NDArray
    ) -> NDArray:
        """Calculate the canopy irradiance.

        Args:
            rho_cb (NDArray): Array of beam irradiances with uniform leaf angle
                distribution.
            I_b (NDArray): Array of beam irradiance values.
            kb_prime (NDArray): Array of scattered beam extinction coefficients.
            I_d (NDArray): Array of diffuse radiation values.

        Returns:
            NDArray: Array of canopy irradiance values.
        """
        I_c = (1 - rho_cb) * I_b * (1 - np.exp(-kb_prime * self.leaf_area_index)) + (
            1 - self.k_rho_cd
        ) * I_d * (1 - np.exp(-kb_prime * self.leaf_area_index))
        return I_c

    def _calc_sunlit_beam_irrad(self, I_b: NDArray, kb: NDArray) -> NDArray:
        """Calculate the sunlit beam irradiance.

        Args:
            I_b (NDArray): Array of beam irradiance values.
            kb (NDArray): Array of beam extinction coefficients.

        Returns:
            NDArray: Array of sunlit beam irradiance values.
        """
        Isun_beam = I_b * (1 - self.k_sigma) * (1 - np.exp(-kb * self.leaf_area_index))
        return Isun_beam

    def _calc_sunlit_diffuse_irrad(self, I_d: NDArray, kb: NDArray) -> NDArray:
        """Calculate the sunlit diffuse irradiance.

        Args:
            I_d (NDArray): Array of diffuse radiation values.
            kb (NDArray): Array of beam extinction coefficients.

        Returns:
            NDArray: Array of sunlit diffuse irradiance values.
        """
        Isun_diffuse = (
            I_d
            * (1 - self.k_rho_cd)
            * (1 - np.exp(-(self.k_kd_prime + kb) * self.leaf_area_index))
            * self.k_kd_prime
            / (self.k_kd_prime + kb)
        )
        return Isun_diffuse

    def _calc_sunlit_scattered_irrad(
        self, I_b: NDArray, rho_cb: NDArray, kb_prime: NDArray, kb: NDArray
    ) -> NDArray:
        """Calculate the sunlit scattered irradiance.

        Args:
            I_b (NDArray): Array of beam irradiance values.
            rho_cb (NDArray): Array of beam irradiances with uniform leaf angle
                distribution.
            kb_prime (NDArray): Array of scattered beam extinction coefficients.
            kb (NDArray): Array of beam extinction coefficients.

        Returns:
            NDArray: Array of sunlit scattered irradiance values.
        """
        Isun_scattered = I_b * (
            (1 - rho_cb)
            * (1 - np.exp(-(kb_prime + kb) * self.leaf_area_index))
            * kb_prime
            / (kb_prime + kb)
            - (1 - self.k_sigma) * (1 - np.exp(-2 * kb * self.leaf_area_index)) / 2
        )
        return Isun_scattered

    def _calc_sunlit_absorbed_irrad(
        self, Isun_beam: NDArray, Isun_diffuse: NDArray, Isun_scattered: NDArray
    ) -> NDArray:
        """Calculate the sunlit absorbed irradiance.

        Args:
            Isun_beam (NDArray): Array of sunlit beam irradiance values.
            Isun_diffuse (NDArray): Array of sunlit diffuse irradiance values.
            Isun_scattered (NDArray): Array of sunlit scattered irradiance values.

        Returns:
            NDArray: Array of sunlit absorbed irradiance values.
        """
        I_csun = Isun_beam + Isun_scattered + Isun_diffuse
        return I_csun

    def _calc_shaded_absorbed_irrad(self, I_c: NDArray, I_csun: NDArray) -> NDArray:
        """Calculate the irradiance absorbed by the shaded fraction of the canopy.

        Args:
            I_c (NDArray): Array of canopy irradiance values.
            I_csun (NDArray): Array of sunlit absorbed irradiance values.

        Returns:
            NDArray: Array of irradiance absorbed by the shaded fraction of the canopy.
        """
        I_cshade = np.where(self.beta_angle > self.k_sol_obs_angle, I_c - I_csun, 0)
        return I_cshade

    def calc_absorbed_irradience(self) -> tuple:
        """Calculate absorbed irradiance for sunlit and shaded leaves.

        Returns:
            tuple: Tuple containing arrays of absorbed irradiance for sunlit
                (I_csun) and shaded (I_cshade) leaves.
        """
        kb = self._calc_beam_extinction_coeff()
        kb_prime = self._calc_scattered_beam_extinction_coeff()
        fd = self._calc_fraction_of_diffuse_rad()
        rho_h = self._calc_beam_irradience_h_leaves()
        rho_cb = self._calc_beam_irrad_unif_leaf_angle_dist(rho_h, kb)
        I_d = self._calc_diffuse_rad(fd)
        I_b = self._calc_beam_irradience(fd)
        I_c = self._calc_canopy_irradience(rho_cb, I_b, kb_prime, I_d)
        Isun_beam = self._calc_sunlit_beam_irrad(I_b, kb)
        Isun_diffuse = self._calc_sunlit_diffuse_irrad(I_d, kb)
        Isun_scattered = self._calc_sunlit_scattered_irrad(I_b, rho_cb, kb_prime, kb)
        I_csun = self._calc_sunlit_absorbed_irrad(
            Isun_beam, Isun_diffuse, Isun_scattered
        )
        I_cshade = self._calc_shaded_absorbed_irrad(I_c, I_csun)

        return I_csun, I_cshade


class TwoLeafAssimilation:
    """Temp."""

    def __init__(
        self, pmodel: PModel | SubdailyPModel, irrad: TwoLeafIrradience, LAI: NDArray
    ):
        self.pmodel = pmodel
        self.irrad = irrad
        self.LAI = LAI

        self.vcmax_pmod: NDArray
        self.vcmax25_pmod: NDArray
        self.optchi_obj: OptimalChiABC
        self.core_const: core_const
        self.gpp: NDArray

        self._get_vcmax_pmod()
        self._get_vcmax_25_pmod()
        self._get_optchi_obj()
        self._get_core_const()

    def _get_vcmax_pmod(self) -> None:
        """Temp."""

        self.vcmax_pmod = (
            self.pmodel.vcmax
            if isinstance(self.pmodel, PModel)
            else self.pmodel.subdaily_vcmax
        )

    def _get_vcmax_25_pmod(self) -> None:
        """Temp."""

        self.vcmax25_pmod = (
            self.pmodel.vcmax25
            if isinstance(self.pmodel, PModel)
            else self.pmodel.subdaily_vcmax25
        )

    def _get_optchi_obj(self) -> None:
        """Temp."""
        self.optchi_obj = (
            self.pmodel.optchi
            if isinstance(self.pmodel, PModel)
            else self.pmodel.optimal_chi
        )

    def _get_core_const(self) -> None:
        """Temp."""
        self.core_const = (
            self.pmodel.core_const
            if isinstance(self.pmodel, PModel)
            else self.pmodel.env.core_const
        )

    def _kv_LLoyd(self) -> NDArray:
        r"""Calculate Kv_Lloyd parameter.

        Generate extinction coefficients to express the vertical gradient in
        photosynthetic capacity after the equation provided in Figure 10 of
        Lloyd et al. (2010)

        :math:`\text{kv\_Lloyd} = \exp(0.00963 \cdot \text{vcmax\_pmod} - 2.43)`

        NB: Vcmax is used here rather than vcmax_25
        """

        kv_Lloyd = np.exp(0.00963 * self.vcmax_pmod - 2.43)

        return kv_Lloyd

    def _Vmax25_canopy(self, kv_Lloyd: NDArray) -> None:
        r"""Calculate carboxylation in the canopy at standard conditions.

        :math:`\text{Vmax25\_canopy} = \text{LAI} \cdot \text{vcmax25\_pmod} \cdot \left
        (\frac{1 - \exp(-\text{kv\_Lloyd})}{\text{kv\_Lloyd}}\right)`
        """

        Vmax25_canopy = (
            self.LAI * self.vcmax25_pmod * ((1 - np.exp(-kv_Lloyd)) / kv_Lloyd)
        )

        return Vmax25_canopy

    def _Vmax25_sun(self, kv_Lloyd: NDArray) -> NDArray:
        r"""Calculate carboxylation in sunlit areas at standard conditions.

        :math:`\text{Vmax25\_sun} = \text{LAI} \cdot \text{vcmax25\_pmod} \cdot \left(\frac{1 -
        \exp(-\text{kv\_Lloyd} - \text{irrad.kb} \cdot \text{LAI})}{\text{kv\_Lloyd} + \text{irrad.kb}
        \cdot \text{LAI}}\right)`
        """

        Vmax25_sun = (
            self.LAI
            * self.vcmax25_pmod
            * (
                (1 - np.exp(-kv_Lloyd - self.irrad.kb * self.LAI))
                / (kv_Lloyd + self.irrad.kb * self.LAI)
            )
        )

        return Vmax25_sun

    def _Vmax25_shade(self, Vmax25_canopy: NDArray, Vmax_25_sun: NDArray) -> NDArray:
        r"""Calculate carboxylation in shade areas at standard conditions."""

        return Vmax25_canopy - Vmax_25_sun

    def _carboxylation_to_T(self, Vmax25: NDArray) -> NDArray:
        r"""Convert carboxylation rates to ambient temperature.

        Convert carboxylation rates to ambient temperature using an Arrhenius function.

        :math:`\text{Vmax\_sun} = \text{Vmax25\} \cdot \exp \left(\frac{64800 \cdot
        (\text{pmodel.env.tc} - 25)}{298 \cdot 8.314 \cdot (\text{pmodel.env.tc} + 273)}
        \right)`

        """

        Vmax = Vmax25 * np.exp(
            64800
            * (self.pmodel.env.tc - 25)
            / (298 * 8.314 * (self.pmodel.env.tc + 273))
        )

        return Vmax

    def _photosynthetic_estimate(self, Vmax: NDArray) -> NDArray:
        """Calculates photosynthetic estimates as V_cmax * mc."""

        Av = Vmax * self.optchi_obj.mc

        return Av

    def _Jmax25(self, Vmax25: NDArray) -> NDArray:
        """Calculates Jmax estimates for sun and shade.

        Uses Eqn 31, after Wullschleger
        """

        Jmax25 = 29.1 + 1.64 * Vmax25

        return Jmax25

    def _Jmax25_temp_correction(self, Jmax25: NDArray) -> NDArray:
        """Temperature correction (Mengoli 2021 Eqn 3b).

        T in K.

        """

        Jmax = Jmax25 * np.exp(
            (43990 / 8.314) * (1 / 298 - 1 / (self.pmodel.env.tc + 273))
        )

        return Jmax

    def _calc_J(
        self, Jmax, I_c: TwoLeafIrradience.I_csun | TwoLeafIrradience.I_cshade
    ) -> NDArray:
        """Calculates J."""

        J = self.Jmax_sun * I_c * (1 - 0.15) / (I_c + 2.2 * Jmax)

        return J

    def _calc_Aj(self, J: NDArray) -> NDArray:
        """Calculate J."""

        Aj = self.optchi_obj.mj * J / 4

        return Aj

    def _Acanopy(self, Aj, Av):
        """Calculate the assimilation in each partition as the minimum of Aj and Av.

        Clip data when the sun is below the angle of obscurity
        """

        ew_minima = np.minimum(Aj, Av)

        Acanopy = np.where(
            self.irrad.beta_angle < self.irrad.solar_obscurity_angle, 0, ew_minima
        )

        return Acanopy

    def _gpp(self, Acanopy_sun: NDArray, Acanopy_shade: NDArray) -> NDArray:
        """Calculate gpp."""

        gpp = self.core_const.k_c_molmass * Acanopy_sun + Acanopy_shade

        return gpp

    def calc_gpp(self) -> None:
        """Calculate gross primary product."""

        kv_Lloyd = self._kv_LLoyd()

        Vmax25_canopy = self._Vmax25_canopy(kv_Lloyd)
        Vmax25_sun = self._Vmax25_sun(kv_Lloyd)
        Vmax25_shade = self._Vmax25_shade(Vmax25_canopy, Vmax25_sun)

        Vmax_sun = self._carboxylation_to_T(Vmax25_sun)
        Vmax_shade = self._carboxylation_to_T(Vmax25_shade)

        Av_sun = self._photosynthetic_estimate(Vmax_sun)
        Av_shade = self._photosynthetic_estimate(Vmax_shade)

        Jmax25_sun = self._Jmax25(Vmax25_sun)
        Jmax25_shade = self._Jmax25(Vmax25_shade)

        Jmax_sun = self._Jmax25_temp_correction(Jmax25_sun)
        Jmax_shade = self._Jmax25_temp_correction(Jmax25_shade)

        J_sun = self._calc_J(Jmax_sun, self.irrad.I_csun)
        J_shade = self._calc_J(Jmax_shade, self.irrad.I_cshade)

        Aj_sun = self._calc_Aj(J_sun)
        Aj_shade = self._calc_Aj(J_shade)

        Acanopy_sun = self._Acanopy(Aj_sun, Av_sun)
        Acanopy_shade = self._Acanopy(Aj_shade, Av_shade)

        gpp = self._gpp(Acanopy_sun, Acanopy_shade)

        return gpp
