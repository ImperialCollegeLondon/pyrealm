"""Test TModel class.

Tests the init, grow_ttree and other methods of TModel.
"""

from importlib import resources

import numpy as np
import pandas as pd
import pytest

# Fixtures: inputs and expected values from the original implementation in R


@pytest.fixture(scope="module")
def rvalues():
    """Fixture to load test inputs from file.

    This is a time series of growth using the default trait values in R, mapped to the
    internal property names used in TTree
    """
    from pyrealm.demography.flora import PlantFunctionalType

    # Load the PFT definitions and rename to pyrealm attributes
    pfts_path = resources.files("pyrealm_build_data.t_model") / "pft_definitions.csv"
    pft_definitions = pd.read_csv(pfts_path)

    pft_definitions = pft_definitions.rename(
        columns={
            "name": "alt_two",
            "d": 0.6,
            "a": 102,
            "cr": 406.12,
            "Hm": 45.33,
            "rho": 100,
            "L": 1.0,
            "sigma": 21,
            "tf": 8,
            "tr": 2.1,
            "K": 0.4,
            "y": 0.7,
            "zeta": 0.15,
            "rr": 0.962,
            "rs": 0.054,
        }
    )

    datapath = resources.files("pyrealm_build_data.t_model") / "rtmodel_output.csv"
    data = pd.read_csv(datapath)

    data = data.rename(
        columns={
            "dD": "delta_d",
            "D": "diameter",
            "H": "height",
            "Ac": "crown_area",
            "Wf": "mass_fol",
            "Ws": "mass_stm",
            "Wss": "mass_swd",
            "GPP": "gpp_actual",
            "Rm1": "resp_swd",
            "Rm2": "resp_frt",
            "dWs": "delta_mass_stm",
            "dWfr": "delta_mass_frt",
        }
    )

    # Get the default PFT traits, which should match the settings used for the
    # regression data set.
    default_pft = PlantFunctionalType()

    # Fix some scaling differences:
    # The R tmodel implementation rescales reported delta_d as a radial increase in
    # millimetres, not diameter increase in metres
    data["delta_d"] = data["delta_d"] / 500

    # The R tmodel implementation slices off foliar respiration costs from GPP before
    # doing anything - the pyrealm.tmodel implementation keeps this cost within the tree
    # calculation, so proportionally inflate the GPP to make it match
    data["gpp_actual"] = data["gpp_actual"] / (1 - default_pft.resp_f)

    return data


def test_calculate_heights(rvalues):
    """Tests happy path for calculation of heights of tree from diameter."""

    from pyrealm.demography.t_model_functions import calculate_heights

    pft_h_max_values = np.array([25.33, 15.33])
    pft_a_hd_values = np.array([116.0, 116.0])
    diameters_at_breast_height = np.array([0.2, 0.6])
    expected_heights = np.array([15.19414157, 15.16639589])
    actual_heights = calculate_heights(
        pft_h_max_values, pft_a_hd_values, diameters_at_breast_height
    )
    assert np.allclose(actual_heights, expected_heights, decimal=8)
