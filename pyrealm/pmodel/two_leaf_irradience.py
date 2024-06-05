"""To do."""

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants.two_leaf_canopy import TwoLeafConst


class TwoLeafIrradience:
    """Calculates two-leaf irradience."""

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
        """To do."""
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
        """To do."""
        kb = np.where(
            self.beta_angle > self.k_sol_obs_angle, 0.5 / np.sin(self.beta_angle), 30
        )

        return kb

    def _calc_scattered_beam_extinction_coeff(self) -> NDArray:
        """To do."""
        kb_prime = np.where(
            self.beta_angle > self.k_sol_obs_angle, 0.46 / np.sin(self.beta_angle), 27
        )

        return kb_prime

    def _calc_fraction_of_diffuse_rad(self) -> NDArray:
        """From deP&F, A23 and A25."""
        m = (self.PATM / self.PA0) / np.sin(self.beta_angle)

        fd = (1 - 0.72**m) / (1 + (0.72**m * (1 / self.k_fa - 1)))

        return fd

    def _calc_beam_irradience_h_leaves(self) -> NDArray:
        """From deP&F A20."""
        rho_h = (1 - np.sqrt(1 - self.k_sigma)) / (1 + np.sqrt(1 - self.k_sigma))

        return rho_h

    def _calc_beam_irrad_unif_leaf_angle_dist(
        self, rho_h: NDArray, kb: NDArray
    ) -> NDArray:
        """From deP&F A19."""

        rho_cb = 1 - np.exp(-2 * rho_h * kb / (1 + kb))

        return rho_cb

    def _calc_diffuse_rad(self, fd: NDArray) -> NDArray:
        """To do."""
        I_d = np.clip(self.PPFD * fd, a_min=0, a_max=np.inf)

        return I_d

    def _calc_beam_irradience(self, fd: NDArray) -> NDArray:
        """To do."""
        I_b = self.PPFD * (1 - fd)

        return I_b

    def _calc_scattered_beam_irradience(
        self, I_b: NDArray, kb: NDArray, kb_prime: NDArray, rho_cb: NDArray
    ) -> NDArray:
        """From deP&F A8."""
        I_bs = I_b * (1 - rho_cb) * kb_prime * np.exp(-kb_prime * self.LAI) - (
            1 - self.k_sigma
        ) * kb * np.exp(-kb * self.LAI)

        return I_bs

    def _calc_canopy_irradience(
        self, rho_cb: NDArray, I_b: NDArray, kb_prime: NDArray, I_d: NDArray
    ) -> NDArray:
        """From deP&F eqn 13."""

        I_c = (1 - rho_cb) * I_b * (1 - np.exp(-kb_prime * self.LAI)) + (
            1 - self.k_rho_cd
        ) * I_d * (1 - np.exp(-kb_prime * self.LAI))

        return I_c

    def _calc_sunlit_beam_irrad(self, I_b: NDArray, kb: NDArray) -> NDArray:
        """From deP&F Eqn 20."""
        Isun_beam = I_b * (1 - self.k_sigma) * (1 - np.exp(-kb * self.LAI))

        return Isun_beam

    def _calc_sunlit_diffuse_irrad(self, I_d: NDArray, kb: NDArray) -> NDArray:
        """To do."""
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
        """deP&F eqn 20."""
        Isun_scattered = I_b * (
            (
                (1 - rho_cb)
                * (1 - np.exp(-(kb_prime + kb) * self.LAI))
                * kb_prime
                / (kb_prime + kb)
            )
            - ((1 - self.k_sigma) * (1 - np.exp(-2 * kb * self.LAI)) / 2)
        )

        return Isun_scattered

    def _calc_sunlit_absorbed_irrad(
        self, Isun_beam: NDArray, Isun_diffuse: NDArray, Isun_scattered: NDArray
    ) -> NDArray:
        """To do."""
        I_csun = Isun_beam + Isun_scattered + Isun_diffuse

        return I_csun

    def _calc_shaded_absorbed_irrad(self, I_c: NDArray, I_csun: NDArray) -> NDArray:
        """Irradiance absorbed by the shaded fraction of the canopy (Eqn 21).

        Including a clause to exclude the hours of obscurity
        """
        I_cshade = np.where(self.beta_angle > self.k_sol_obs_angle, I_c - I_csun, 0)

        return I_cshade

    def calc_absorbed_irradience(self) -> tuple:
        """To do."""
        kb = self._calc_beam_extinction_coeff()

        kb_prime = self._calc_scattered_beam_extinction_coeff()

        fd = self._calc_fraction_of_diffuse_rad()

        rho_h = self._calc_beam_irradience_h_leaves()

        rho_cb = self._calc_beam_irrad_unif_leaf_angle_dist(rho_h, kb)

        I_d = self._calc_diffuse_rad(fd)

        I_b = self._calc_beam_irradience(fd)

        # ßI_bs = self._calc_scattered_beam_irradience(I_b, kb, kb_prime)

        I_c = self._calc_canopy_irradience(rho_cb, I_b, kb_prime, I_d)

        Isun_beam = self._calc_sunlit_beam_irrad(I_b, kb)

        Isun_diffuse = self._calc_sunlit_diffuse_irrad(I_d, kb)

        Isun_scattered = self._calc_sunlit_scattered_irrad(I_b, rho_cb, kb_prime, kb)

        I_csun = self._calc_sunlit_absorbed_irrad(
            Isun_beam, Isun_diffuse, Isun_scattered
        )

        I_cshade = self._calc_shaded_absorbed_irrad(I_c, I_csun)

        return I_csun, I_cshade


def beta_angle(lat: NDArray, d: NDArray, h: NDArray) -> NDArray:
    """Calculates solar beta angle.

    Calculates solar beta angle using Eq A13 of dePury & Farquhar (1997).

    Args:
        lat: array of latitudes (rads)
        d: array of declinations (rads)
        h: array of hour angle (rads)

    Returns:
        beta: array of solar beta angles
    """

    beta = np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(h)

    return beta
