"""Regression tests of the demography.t_model functions.

The original R implementation has been used to generate a set of predicted growth
trajectories across a set of PFT definitions (default from the original paper and then
two fairly randomly chosen variants).
"""

from importlib import resources

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

# Fixtures: inputs and expected values from the original implementation in R


@pytest.fixture(scope="module")
def rvalues():
    """Fixture to load test inputs from file.

    The regression test inputs consist of time series of growth from an initial DBH
    run using the original R implementation of the T model, for each of a small set of
    different plant functional type definitions. The PFT definitions are loaded first
    and then the output file associated with each PFT is loaded.
    """

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

    # Convert to dicts for creating PFT instances
    pft_definitions = pft_definitions.to_dict(orient="records")

    return_value = []

    # Loop over the PFT definitions
    for pft in pft_definitions:
        # Record the starting DBH and name
        dbh_init = pft.pop("d")
        pft_name = pft.pop("name")

        # Add foliar respiration
        pft["resp_f"] = 0.1

        # Convert dict values to row arrays
        pft = {k: np.array([v]) for k, v in pft.items()}

        # Load the appropriate output file and then remap the field names
        datapath = (
            resources.files("pyrealm_build_data.t_model")
            / f"rtmodel_output_{pft_name}.csv"
        )
        data = pd.read_csv(datapath)

        data = data.rename(
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
        data["delta_d"] = data["delta_d"] / 500

        # NOTE: The reported P0 in the R tmodel outputs has already had fixed foliar
        # respiration costs removed before calculating anything. The pyrealm
        # implementation has this as a PFT trait, so in some tests the potential GPP
        # will need to be proportionally scaled up to make them match, but this is not
        # true for _all_ tests so the values here are left untouched.

        # Convert data to column arrays
        data = data.to_dict(orient="list")
        data = {k: np.array(v)[:, None] for k, v in data.items()}

        # Add a tuple of the inputs and outputs to the return list.
        return_value.append((pft, dbh_init, data))

    return return_value


def test_calculate_heights(rvalues):
    """Test calculation of heights of tree from diameter."""

    from pyrealm.demography.t_model_functions import calculate_heights

    for pft, _, data in rvalues:
        actual_heights = calculate_heights(
            h_max=pft["h_max"], a_hd=pft["a_hd"], dbh=data["diameter"]
        )

        assert_array_almost_equal(actual_heights, data["height"], decimal=8)


def test_calculate_crown_areas(rvalues):
    """Tests calculation of crown areas of trees."""
    from pyrealm.demography.t_model_functions import calculate_crown_areas

    for pft, _, data in rvalues:
        actual_crown_areas = calculate_crown_areas(
            ca_ratio=pft["ca_ratio"],
            a_hd=pft["a_hd"],
            dbh=data["diameter"],
            stem_height=data["height"],
        )

        assert_array_almost_equal(actual_crown_areas, data["crown_area"], decimal=8)


def test_calculate_crown_fractions(rvalues):
    """Tests calculation of crown fractions of trees."""

    from pyrealm.demography.t_model_functions import calculate_crown_fractions

    for pft, _, data in rvalues:
        actual_crown_fractions = calculate_crown_fractions(
            a_hd=pft["a_hd"],
            dbh=data["diameter"],
            stem_height=data["height"],
        )
        assert_array_almost_equal(
            actual_crown_fractions, data["crown_fraction"], decimal=8
        )


def test_calculate_stem_masses(rvalues):
    """Tests happy path for calculation of stem masses of trees."""

    from pyrealm.demography.t_model_functions import calculate_stem_masses

    for pft, _, data in rvalues:
        actual_stem_masses = calculate_stem_masses(
            dbh=data["diameter"],
            stem_height=data["height"],
            rho_s=pft["rho_s"],
        )
        assert_array_almost_equal(actual_stem_masses, data["mass_stm"], decimal=8)


def test_calculate_foliage_masses(rvalues):
    """Tests calculation of foliage masses of trees."""

    from pyrealm.demography.t_model_functions import calculate_foliage_masses

    for pft, _, data in rvalues:
        actual_foliage_masses = calculate_foliage_masses(
            crown_area=data["crown_area"], lai=pft["lai"], sla=pft["sla"]
        )
        assert_array_almost_equal(actual_foliage_masses, data["mass_fol"], decimal=8)


def test_calculate_sapwood_masses(rvalues):
    """Tests calculation of sapwood masses of trees."""

    from pyrealm.demography.t_model_functions import calculate_sapwood_masses

    for pft, _, data in rvalues:
        actual_sapwood_masses = calculate_sapwood_masses(
            crown_area=data["crown_area"],
            stem_height=data["height"],
            crown_fraction=data["crown_fraction"],
            ca_ratio=pft["ca_ratio"],
            rho_s=pft["rho_s"],
        )
        assert_array_almost_equal(actual_sapwood_masses, data["mass_swd"], decimal=8)


def test_calculate_whole_crown_gpp(rvalues):
    """Tests calculation of sapwood masses of trees.

    Note that this test can used reported P0 from R directly - no need to correct for
    foliar respiration.
    """

    from pyrealm.demography.t_model_functions import calculate_whole_crown_gpp

    for pft, _, data in rvalues:
        actual_whole_crown_gpp = calculate_whole_crown_gpp(
            potential_gpp=data["potential_gpp"],
            crown_area=data["crown_area"],
            par_ext=pft["par_ext"],
            lai=pft["lai"],
        )
        assert_array_almost_equal(actual_whole_crown_gpp, data["crown_gpp"], decimal=8)


def test_calculate_sapwood_respiration(rvalues):
    """Tests calculation of sapwood respiration of trees."""

    from pyrealm.demography.t_model_functions import calculate_sapwood_respiration

    for pft, _, data in rvalues:
        actual_sapwood_respiration = calculate_sapwood_respiration(
            sapwood_mass=data["mass_swd"],
            resp_s=pft["resp_s"],
        )
        assert_array_almost_equal(
            actual_sapwood_respiration, data["resp_swd"], decimal=8
        )


def test_calculate_foliar_respiration(rvalues):
    """Tests calculation of foliar respiration of trees.

    This is implemented as a fixed proportion of GPP - and the reported values from R
    are automatically penalised by this proportion beforehand, so this test looks
    circular but is important to validate this difference.
    """

    from pyrealm.demography.t_model_functions import calculate_foliar_respiration

    for pft, _, data in rvalues:
        actual_foliar_respiration = calculate_foliar_respiration(
            whole_crown_gpp=data["crown_gpp"],
            resp_f=pft["resp_f"],
        )
        assert_array_almost_equal(
            actual_foliar_respiration,
            data["crown_gpp"] * pft["resp_f"],
            decimal=8,
        )


def test_calculate_fine_root_respiration(rvalues):
    """Tests calculation of fine root respiration of trees."""

    from pyrealm.demography.t_model_functions import calculate_fine_root_respiration

    for pft, _, data in rvalues:
        actual_fine_root_respiration = calculate_fine_root_respiration(
            zeta=pft["zeta"],
            sla=pft["sla"],
            resp_r=pft["resp_r"],
            foliage_mass=data["mass_fol"],
        )
        assert_array_almost_equal(
            actual_fine_root_respiration,
            data["resp_frt"],
            decimal=8,
        )


def test_calculate_net_primary_productivity(rvalues):
    """Tests calculation of fine root respiration of trees.

    Again - this test has to account for the R implementation removing foliar
    respiration from potential GPP before calculating crown GPP.
    """

    from pyrealm.demography.t_model_functions import calculate_net_primary_productivity

    for pft, _, data in rvalues:
        actual_npp = calculate_net_primary_productivity(
            yld=pft["yld"],
            whole_crown_gpp=data["crown_gpp"] / (1 - pft["resp_f"]),
            foliar_respiration=data["crown_gpp"] / (1 - pft["resp_f"]) * pft["resp_f"],
            fine_root_respiration=data["resp_frt"],
            sapwood_respiration=data["resp_swd"],
        )
        assert_array_almost_equal(
            actual_npp,
            data["NPP"],
            decimal=8,
        )


def test_calculate_foliage_and_fine_root_turnover(rvalues):
    """Tests calculation of fine root respiration of trees."""

    from pyrealm.demography.t_model_functions import (
        calculate_foliage_and_fine_root_turnover,
    )

    for pft, _, data in rvalues:
        actual_turnover = calculate_foliage_and_fine_root_turnover(
            sla=pft["sla"],
            tau_f=pft["tau_f"],
            zeta=pft["zeta"],
            tau_r=pft["tau_r"],
            foliage_mass=data["mass_fol"],
        )
        assert_array_almost_equal(
            actual_turnover,
            data["turnover"],
            decimal=8,
        )


def test_calculate_growth_increments(rvalues):
    """Tests calculation of fine root respiration of trees."""

    from pyrealm.demography.t_model_functions import (
        calculate_growth_increments,
    )

    for pft, _, data in rvalues:
        delta_dbh, delta_mass_stem, delta_mass_fine_root = calculate_growth_increments(
            rho_s=pft["rho_s"],
            a_hd=pft["a_hd"],
            h_max=pft["h_max"],
            lai=pft["lai"],
            ca_ratio=pft["ca_ratio"],
            sla=pft["sla"],
            zeta=pft["zeta"],
            npp=data["NPP"],
            turnover=data["turnover"],
            dbh=data["diameter"],
            stem_height=data["height"],
        )
        assert_array_almost_equal(
            delta_dbh,
            data["delta_d"],
            decimal=8,
        )
        assert_array_almost_equal(
            delta_mass_stem,
            data["delta_mass_stm"],
            decimal=8,
        )
        assert_array_almost_equal(
            delta_mass_fine_root,
            data["delta_mass_frt"],
            decimal=8,
        )
