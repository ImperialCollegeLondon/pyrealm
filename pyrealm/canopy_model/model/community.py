"""placeholder."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import pyrealm.canopy_model.functions.jaideep_t_model_extension_functions as t_model_extension
from pyrealm import t_model_functions as t_model
from pyrealm.canopy_model.model.flora import Flora, PlantFunctionalType


class Community:
    """placeholder."""

    def __init__(
        self,
        cell_id: int,
        cell_area: float,
        cohort_dbh_values: NDArray[np.float32],
        cohort_number_of_individuals: NDArray[np.int_],
        cohort_pft_names: NDArray[np.str_],
        flora: Flora,
    ):
        # community wide properties
        self.cell_id: int = cell_id
        self.cell_area: float = cell_area
        self.flora: Flora = flora
        self.__number_of_cohorts: int = len(cohort_dbh_values)

        # arrays representing properties of cohorts
        self.cohort_dbh_values: NDArray[np.float32] = cohort_dbh_values
        self.cohort_number_of_individuals: NDArray[np.int_] = (
            cohort_number_of_individuals
        )
        self.cohort_pft_names: NDArray[np.str_] = cohort_pft_names

        # initialise empty arrays representing properties of plant functional types
        self.pft_a_hd_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_ca_ratio_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_h_max_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_lai_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_par_ext_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_resp_f_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_resp_r_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_resp_s_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_rho_s_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_sla_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_tau_f_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_tau_r_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_yld_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_zeta_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_m_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )
        self.pft_n_values: NDArray[np.float32] = np.empty(
            self.__number_of_cohorts, dtype=np.float32
        )

        self.__populate_pft_arrays()

        # Create arrays containing values relating to T Model geometry
        self.t_model_heights = t_model.calculate_heights(
            self.pft_h_max_values, self.pft_a_hd_values, self.cohort_dbh_values
        )

        self.t_model_crown_areas = t_model.calculate_crown_areas(
            self.pft_ca_ratio_values,
            self.pft_a_hd_values,
            self.cohort_dbh_values,
            self.t_model_heights,
        )

        self.t_model_crown_fractions = t_model.calculate_crown_fractions(
            self.t_model_heights, self.pft_a_hd_values, self.cohort_dbh_values
        )

        self.t_model_stem_masses = t_model.calculate_stem_masses(
            self.cohort_dbh_values, self.t_model_heights, self.pft_rho_s_values
        )

        self.t_model_foliage_masses = t_model.calculate_foliage_masses(
            self.t_model_crown_areas, self.pft_lai_values, self.pft_sla_values
        )

        self.t_model_swd_masses = t_model.calculate_swd_masses(
            self.t_model_crown_areas,
            self.pft_rho_s_values,
            self.t_model_heights,
            self.t_model_crown_fractions,
            self.pft_ca_ratio_values,
        )

        # Create arrays containing properties pertaining to Jaideep's t model
        # extension
        self.canopy_factor_q_m_values = t_model_extension.calculate_q_m_values(
            self.pft_m_values, self.pft_n_values
        )
        self.canopy_factor_r_0_values = t_model_extension.calculate_r_0_values(
            self.canopy_factor_q_m_values, self.t_model_crown_areas
        )
        self.canopy_factor_z_m_values = t_model_extension.calculate_z_m_values(
            self.pft_m_values, self.pft_n_values, self.t_model_heights
        )

    @classmethod
    def load_communities_from_csv(
        cls, cell_area: float, csv_path: str, flora: Flora
    ) -> list[Community]:

        community_data = pd.read_csv(csv_path)

        data_grouped_by_community = community_data.groupby(community_data.cell_id)

        communities = []

        for cell_id in data_grouped_by_community.groups:
            community_dataframe = data_grouped_by_community.get_group(cell_id)
            dbh_values = community_dataframe["dbh"].to_numpy(dtype=np.float32)
            number_of_individuals = community_dataframe["n"].to_numpy(dtype=np.int_)
            pft_names = community_dataframe["pft"].to_numpy(dtype="U20")
            community_object = Community(cell_id,
                                         cell_area,
                                         dbh_values,
                                         number_of_individuals,
                                         pft_names,
                                         flora)
            communities.append(community_object)

        return communities

    def __populate_pft_arrays(self) -> None:
        """Populate plant functional type arrays.

        Populate the initialised arrays containing properties relating to plant
        functional type by looking up the plant functional type in the Flora dictionary,
        extracting the properties and inserting them into the relevant position in the
        array.
        :return: None
        """
        for i in range(0, self.__number_of_cohorts - 1):
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

    def __look_up_plant_functional_type(self, pft_name: str) -> PlantFunctionalType:
        """Retrieve plant functional type for a cohort from the flora dictionary."""
        pft = self.flora.get(pft_name)

        if pft is None:
            raise Exception(f"Cohort data supplied with in an invalid PFT name: {pft_name}")
        return pft