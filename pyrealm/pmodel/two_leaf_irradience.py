"""To do."""

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants.two_leaf_canopy import TwoLeafConst


class TwoLeafIrradience:
    """Calculates daytime and nighttime net radiation using the two-leaf model.

    Args:
        beta_angle (NDArray): Array of beta angles (radians).
        PPFD (NDArray): Array of photosynthetic photon flux density values.
        LAI (NDArray): Array of leaf area index values.
        PATM (NDArray): Array of atmospheric pressure values.
    """

    def __init__(self, beta_angle: NDArray, PPFD: NDArray, LAI: NDArray, PATM: NDArray):
        self.beta_angle = beta_angle
        self.PPFD = PPFD
        self.LAI = LAI
        self.PATM = PATM

        constants = TwoLeafConst()

        self.PA0: float = constants.k_PA0
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
            arrays = [self.beta_angle, self.PPFD, self.LAI, self.PATM]
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
        m = (self.PATM / self.PA0) / np.sin(self.beta_angle)
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
        I_d = np.clip(self.PPFD * fd, a_min=0, a_max=np.inf)
        return I_d

    def _calc_beam_irradience(self, fd: NDArray) -> NDArray:
        """Calculate the beam irradiance.

        Args:
            fd (NDArray): Array of fractions of diffuse radiation.

        Returns:
            NDArray: Array of beam irradiance values.
        """
        I_b = self.PPFD * (1 - fd)
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
        I_bs = I_b * (1 - rho_cb) * kb_prime * np.exp(-kb_prime * self.LAI) - (
            1 - self.k_sigma
        ) * kb * np.exp(-kb * self.LAI)
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
        I_c = (1 - rho_cb) * I_b * (1 - np.exp(-kb_prime * self.LAI)) + (
            1 - self.k_rho_cd
        ) * I_d * (1 - np.exp(-kb_prime * self.LAI))
        return I_c

    def _calc_sunlit_beam_irrad(self, I_b: NDArray, kb: NDArray) -> NDArray:
        """Calculate the sunlit beam irradiance.

        Args:
            I_b (NDArray): Array of beam irradiance values.
            kb (NDArray): Array of beam extinction coefficients.

        Returns:
            NDArray: Array of sunlit beam irradiance values.
        """
        Isun_beam = I_b * (1 - self.k_sigma) * (1 - np.exp(-kb * self.LAI))
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
            * (1 - np.exp(-(self.k_kd_prime + kb) * self.LAI))
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
            * (1 - np.exp(-(kb_prime + kb) * self.LAI))
            * kb_prime
            / (kb_prime + kb)
            - (1 - self.k_sigma) * (1 - np.exp(-2 * kb * self.LAI)) / 2
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


def beta_angle(latitude: NDArray, declination: NDArray, hour_angle: NDArray) -> NDArray:
    """Calculates solar beta angle.

    Calculates solar beta angle using Eq A13 of dePury & Farquhar (1997).

    Args:
        latitude: array of latitudes (rads)
        declination: array of declinations (rads)
        hour_angle: array of hour angle (rads)

    Returns:
        beta: array of solar beta angles
    """

    beta = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(
        declination
    ) * np.cos(hour_angle)

    return beta
