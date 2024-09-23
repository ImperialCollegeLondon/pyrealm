"""Shared fixtures for testing the demography module."""

from importlib import resources

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rtmodel_flora():
    """Generates a flora object from the rtmodel test definitions."""

    from pyrealm.demography.flora import Flora, PlantFunctionalType

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

    pft_definitions = pft_definitions.drop(columns=["d"])

    return Flora(
        pfts=[
            PlantFunctionalType(**args)
            for args in pft_definitions.to_dict(orient="records")
        ]
    )


@pytest.fixture()
def rtmodel_flora_dict(rtmodel_flora):
    """Returns the data dictionary attribute of the rtmodel Flora object."""
    return rtmodel_flora.data


@pytest.fixture
def rtmodel_data():
    """Loads some simple predictions from the R implementation for testing."""

    rdata_path = (
        resources.files("pyrealm_build_data.t_model") / "rtmodel_unit_testing.csv"
    )
    rdata = pd.read_csv(rdata_path)

    rdata = rdata.rename(
        columns={
            "dD": "delta_d",
            "D": "dbh",
            "H": "stem_height",
            "fc": "crown_fraction",
            "Ac": "crown_area",
            "Wf": "foliage_mass",
            "Ws": "stem_mass",
            "Wss": "sapwood_mass",
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
    rdata_arrays = {k: np.reshape(v, (3, 6)).T for k, v in rdata.items()}

    return rdata_arrays
