"""Shared fixtures for demography testing."""

from importlib import resources

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rtmodel_flora():
    """Generates a flora object from the rtmodel test definitions."""

    from pyrealm.demography.flora import create_flora

    # Load the PFT definitions and rename to pyrealm attributes
    pfts_path = resources.files("pyrealm_build_data.t_model") / "pft_definitions.csv"
    pft_definitions = pd.read_csv(pfts_path)

    # Map the PFT trait args from the R implementation to pyrealm
    pft_definitions = pft_definitions.rename(
        columns={
            "name": "pft_name",
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

    # Set foliar respiration to zero to avoid issues with this being applied before
    # estimating whole crown gpp in rtmodel
    pft_definitions["resp_f"] = 0

    # Set reproductive tissue parameters to zero as they are not in the original T model
    pft_definitions["resp_rt"] = 0
    pft_definitions["tau_rt"] = 1
    pft_definitions["p_foliage_for_reproductive_tissue"] = 0
    pft_definitions["gpp_topslice"] = 0

    return create_flora(**pft_definitions.to_dict(orient="list"))


@pytest.fixture
def rtmodel_data():
    """Loads some simple predictions from the R implementation for testing."""

    rdata_path = (
        resources.files("pyrealm_build_data.t_model") / "rtmodel_unit_testing.csv"
    )
    rdata = pd.read_csv(rdata_path)

    rdata = rdata.rename(
        columns={
            "D": "dbh",
            "H": "stem_height",
            "fc": "crown_fraction",
            "Ac": "crown_area",
            "Wf": "foliage_mass",
            "Ws": "stem_mass",
            "Wss": "sapwood_mass",
            "P0": "potential_gpp",
            "GPP": "whole_crown_gpp",
            "Rm1": "sapwood_respiration",
            "Rm2": "fine_root_respiration",
            "NPP": "npp",
            "dD": "delta_dbh",
            "dWs": "delta_stem_mass",
            "dWfr": "delta_foliage_mass",
        }
    )

    # Fix some scaling differences:
    # The R tmodel implementation rescales reported delta_d as a radial increase in
    # millimetres, not diameter increase in metres
    rdata["delta_dbh"] = rdata["delta_dbh"] / 500

    # Wrap the return data into arrays with PFT as columns and diameter values as rows
    rdata_arrays = {k: np.reshape(v, (3, 6)).T for k, v in rdata.items()}

    return rdata_arrays


@pytest.fixture
def fixture_cohorts_and_allometry():
    """A fixture providing a simple community."""

    from pyrealm.demography.cohorts import cohort_id_generator, create_cohorts
    from pyrealm.demography.flora import create_flora
    from pyrealm.demography.tmodel import StemAllometry

    # A simple community containing one sample stem, with an initial crown gap fraction
    # of zero.
    flora = create_flora(dict(pft_name=("test",), f_g=(0.0,)))
    cid_gen = cohort_id_generator()
    cohorts = create_cohorts(
        pft_name=np.array(["test"] * 4),
        dbh_value=np.array([0.2, 0.4, 0.6, 0.8]),
        n_individuals=np.array([1] * 4),
        flora=flora,
        cid_generator=cid_gen,
    )
    allometry = StemAllometry(cohorts=cohorts)

    return cohorts, allometry
