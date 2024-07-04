"""placeholder."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pyrealm.canopy_model.model.flora import Flora, PlantFunctionalType


class Community:
    """placeholder."""

    def __init__(
        self,
        cell_id: int,
        cell_area: float,
        cohort_dbh_values: NDArray[np.float32],
        cohort_number_of_individuals_values: NDArray[np.int_],
        cohort_pft_names: NDArray[np.str_],
        flora: Flora,
    ):
        # community wide properties
        self.cell_id: int = cell_id
        self.cell_area: float = cell_area
        self.flora: Flora = flora
        self.number_of_cohorts: int = len(cohort_dbh_values)

        # arrays representing properties of cohorts
        self.cohort_dbh_values: NDArray[np.float32] = cohort_dbh_values
        self.cohort_number_of_individuals_values: NDArray[np.int_] = (
            cohort_number_of_individuals_values
        )
        self.cohort_pft_names: NDArray[np.str_] = cohort_pft_names

        # initialise empty arrays representing properties of plant functional types
        self.pft_a_hd_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_ca_ratio_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_h_max_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_lai_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_par_ext_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_resp_f_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_resp_r_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_resp_s_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_rho_s_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_sla_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_tau_f_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_tau_r_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_yld_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_zeta_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_m_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.pft_n_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )

        # initialise empty arrays representing properties calculated using the t model
        self.t_model_heights: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.t_model_crown_areas: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.t_model_crown_fractions: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.t_model_stem_masses: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.t_model_foliage_masses: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.t_model_swd_masses: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )

        # initialise empty arrays containing properties pertaining to Jaideep's t model
        # extension
        self.canopy_factor_q_m_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.canopy_factor_z_m_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )
        self.canopy_factor_r_0_values: NDArray[np.float32] = np.empty(
            self.number_of_cohorts, dtype=np.float32
        )

        # populate the initialised arrays with the relevant calculations
        self.__populate_pft_arrays()
        self.__calculate_t_model_geometry_arrays()
        self.__calculate_canopy_factor_arrays()

    # Note: These functions have a lot of side effects and this is all very stateful
    def __populate_pft_arrays(self) -> None:
        """Populate plant functional type arrays.

        Populate the initialised arrays containing properties relating to plant
        functional type by looking up the plant functional type in the Flora dictionary,
        extracting the properties and inserting them into the relevant position in the
        array.
        :return: None
        """
        for i in range(0, self.number_of_cohorts - 1):
            pft = self.__look_up_plant_functional_type(self.cohort_pft_names[i])
            self.pft_a_hd_values[i] = pft.a_hd
            self.pft_ca_ratio_values[i] = pft.ca_ratio
            self.pft_h_max_values[i] = pft.h_max
            self.pft_lai_values[i] = pft.lai
            self.pft_par_ext_values[i] = pft.par_ext
            self.pft_resp_f_values[i] = pft.resp_f
            self.pft_resp_r_values[i] = pft.resp_r
            self.pft_resp_s_values[i] = pft.resp_s
            self.pft_rho_s_values[i] = pft.rho_s
            self.pft_sla_values[i] = pft.sla
            self.pft_tau_f_values[i] = pft.tau_f
            self.pft_tau_r_values[i] = pft.tau_r
            self.pft_yld_values[i] = pft.yld
            self.pft_zeta_values[i] = pft.zeta
            self.pft_m_values[i] = pft.m
            self.pft_n_values[i] = pft.n

    def __calculate_t_model_geometry_arrays(self) -> None:
        """Populate t model arrays.

        Populate the relevant initialised arrays with properties calculated using the
        t model.
        :return: None
        """
        self.t_model_heights = self.pft_h_max_values * (
            1
            - np.exp(
                -self.pft_a_hd_values * self.cohort_dbh_values / self.pft_h_max_values
            )
        )

        # Crown area of tree, Equation (8) of Li ea.
        self.t_model_crown_areas = (
            (np.pi * self.pft_ca_ratio_values / (4 * self.pft_a_hd_values))
            * self.cohort_dbh_values
            * self.t_model_heights
        )

        # Crown fraction, Equation (11) of Li ea.
        self.t_model_crown_fractions = self.t_model_heights / (
            self.pft_a_hd_values * self.cohort_dbh_values
        )

        # Masses
        self.t_model_stem_masses = (
            (np.pi / 8)
            * (self.cohort_dbh_values**2)
            * self.t_model_heights
            * self.pft_rho_s_values
        )
        self.t_model_foliage_masses = (
            self.t_model_crown_areas * self.pft_lai_values * (1 / self.pft_sla_values)
        )
        self.t_model_swd_masses = (
            self.t_model_crown_areas
            * self.pft_rho_s_values
            * self.t_model_heights
            * (1 - self.t_model_crown_fractions / 2)
            / self.pft_ca_ratio_values
        )

    def __calculate_canopy_factor_arrays(self) -> None:
        """Populate canopy factor arrays.

        Populate the relevant initialised arrays with properties calculated using
        Jaideep's extension to the T Model.
        :return:
        """
        pass

    def __look_up_plant_functional_type(self, pft_name: str) -> PlantFunctionalType:
        """Retrieve plant functional type for a cohort from the flora dictionary."""
        pft = self.flora.get(pft_name)

        if pft is None:
            raise Exception("Cohort data supplied with in an invalid PFT name.")
        return pft
