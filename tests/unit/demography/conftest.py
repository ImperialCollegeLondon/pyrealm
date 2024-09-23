"""Shared fixtures for testing the demography module."""

from importlib import resources

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rtmodel_data():
    """Loads some simple predictions from the R implementation for testing."""

    # Load the PFT definitions and rename to pyrealm attributes
    pfts_path = resources.files("pyrealm_build_data.t_model") / "pft_definitions.csv"
    pft_definitions = pd.read_csv(pfts_path)

    # Map the PFT trait args from the R implementation to pyrealm
    pft_definitions = pft_definitions.rename(
        columns={
            "a": "a_hd",
            "cr": "ca_ratio",
            "Hm": "h_max",
            "rho": "rho_s",
            "L": "lai",
            "sigma": "sla",
            "tf": "tau_f",
            "tr": "tau_r",
            "K": "par_ext",
            "y": "yld",
            "rr": "resp_r",
            "rs": "resp_s",
        }
    )

    # Add foliar respiration rate as 0.1, as this is handled outside of the R
    # implementation as a function of GPP.
    pft_definitions["resp_f"] = 0.1

    rdata_path = (
        resources.files("pyrealm_build_data.t_model") / "rtmodel_unit_testing.csv"
    )
    rdata = pd.read_csv(rdata_path)

    rdata = rdata.rename(
        columns={
            "dD": "delta_d",
            "D": "diameter",
            "H": "height",
            "fc": "crown_fraction",
            "Ac": "crown_area",
            "Wf": "mass_fol",
            "Ws": "mass_stm",
            "Wss": "mass_swd",
            "P0": "potential_gpp",
            "GPP": "crown_gpp",
            "Rm1": "resp_swd",
            "Rm2": "resp_frt",
            "dWs": "delta_mass_stm",
            "dWfr": "delta_mass_frt",
        }
    )

    # Fix some scaling differences:
    # The R tmodel implementation rescales reported delta_d as a radial increase in
    # millimetres, not diameter increase in metres
    rdata["delta_d"] = rdata["delta_d"] / 500

    # Wrap the return data into arrays with PFT as columns and diameter values as rows
    pft_arrays = {k: v.to_numpy() for k, v in pft_definitions.items()}
    rdata_arrays = {k: np.reshape(v, (3, 6)).T for k, v in rdata.items()}

    return pft_arrays, rdata_arrays
